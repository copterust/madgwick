//! Madgwick's orientation filter
//!
//! This implementation of Madgwick's orientation filter computes 3D orientation from Magnetic,
//! Angular Rate and Gravity (MARG) sensor data and it's currently missing gyroscope drift
//! compensation. (see figure 3 of the report; group 2 has not been implemented)
//!
//! # References
//!
//! - [Madgwick's internal report](http://x-io.co.uk/res/doc/madgwick_internal_report.pdf)

#![deny(missing_docs)]
#![deny(warnings)]
#![no_std]

extern crate libm;
#[macro_use]
extern crate mat;

use core::ops;

use libm::F32Ext;
use mat::traits::{Matrix, Transpose};

/// MARG orientation filter
pub struct Marg {
    beta: f32,
    q: Quaternion,
    sample_period: f32,
}

impl Marg {
    /// Creates a new MARG filter
    ///
    /// - `beta`, filter gain. See section 3.6 of the report for details.
    /// - `sample_period`, period at which the sensors are being sampled (unit: s)
    pub fn new(beta: f32, sample_period: f32) -> Self {
        Marg {
            beta,
            q: Quaternion(1.0, 0.0, 0.0, 0.0),
            sample_period,
        }
    }

    /// Updates the IMU filter and returns the current estimate
    /// - `g`, gyroscope readings
    /// - `a`, accelerometer readings
    pub fn update_imu(&mut self, g: F32x3, a: F32x3) -> Quaternion {
	    // Rate of change of quaternion from gyroscope
        let mut q_dot = Quaternion(
            0.5 * (-self.q.1 * g.x - self.q.2 * g.y - self.q.3 * g.z),
	        0.5 * ( self.q.0 * g.x + self.q.2 * g.z - self.q.3 * g.y),
	        0.5 * ( self.q.0 * g.y - self.q.1 * g.z + self.q.3 * g.x),
	        0.5 * ( self.q.0 * g.z + self.q.1 * g.y - self.q.2 * g.x)
        );

        let mut a = a;
        if !((0.0 == a.x) && (0.0 == a.y) && (0.0 == a.z)) {
            // Normalise accelerometer measurement
            a *= rsqrt(a.norm());
            let q2 = Quaternion(
                2.0 * self.q.0,
                2.0 * self.q.1,
                2.0 * self.q.2,
                2.0 * self.q.3);

            let q0_4 = 4.0 * self.q.0;
            let q1_4 = 4.0 * self.q.1;
            let q2_4 = 4.0 * self.q.2;
            let q1_8 = 8.0 * self.q.1;
            let q2_8 = 8.0 * self.q.2;

            let q0q0 = self.q.0 * self.q.0;
            let q1q1 = self.q.1 * self.q.1;
            let q2q2 = self.q.2 * self.q.2;
            let q3q3 = self.q.3 * self.q.3;

            let mut s0 = q0_4 * q2q2 + q2.2 * a.x + q0_4 * q1q1 - q2.1 * a.y;
            let mut s1 = q1_4 * q3q3 - q2.3 * a.x + 4.0 * q0q0 * self.q.1 - q2.0 * a.y - q1_4 + q1_8 * q1q1 + q1_8 * q2q2 + q1_4 * a.z;
            let mut s2 = 4.0 * q0q0 * self.q.2 + q2.0 * a.x + q2_4 * q3q3 - q2.3 * a.y - q2_4 + q2_8 * q1q1 + q2_8 * q2q2 + q2_4 * a.z;
            let mut s3 = 4.0 * q1q1 * self.q.3 - q2.1 * a.x + 4.0 * q2q2 * self.q.3 - q2.2 * a.y;

            let recip_norm = rsqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
            s0 *= recip_norm;
            s1 *= recip_norm;
            s2 *= recip_norm;
            s3 *= recip_norm;

            q_dot.0 -= self.beta * s0;
            q_dot.1 -= self.beta * s1;
            q_dot.2 -= self.beta * s2;
            q_dot.3 -= self.beta * s3;
        }

        self.q.0 += q_dot.0 * self.sample_period;
        self.q.1 += q_dot.1 * self.sample_period;
        self.q.2 += q_dot.2 * self.sample_period;
        self.q.3 += q_dot.3 * self.sample_period;

        // normalize the quaternion
        self.q *= rsqrt(self.q.norm());

        self.q
    }

    /// Updates the MARG filter and returns the current estimate of the 3D orientation
    ///
    /// - `m`, magnetic north / magnetometer readings
    /// - `ar`, angular rate / gyroscope readings (unit: rad / s)
    /// - `g`, gravity vector / accelerometer readings
    // This implements the block diagram in figure 3, minus the gyroscope drift compensation
    pub fn update(&mut self, mut m: F32x3, ar: F32x3, g: F32x3) -> Quaternion {
        // the report calls the accelerometer reading `a`; let's follow suit
        let mut a = g;

        // vector of angular rates
        let omega = Quaternion(0., ar.x, ar.y, ar.z);

        // rate of change of quaternion from gyroscope (Eq 11)
        let mut dqdt = 0.5 * self.q * omega;

        // normalize orientation vectors
        a *= rsqrt(a.norm());
        m *= rsqrt(m.norm());

        // direction of the earth's magnetic field (Eq. 45 & 46)
        let h = self.q * Quaternion(0., m.x, m.y, m.z) * self.q.conj();
        let bx = (h.1 * h.1 + h.2 * h.2).sqrt();
        let bz = h.3;

        // gradient descent
        let q1 = self.q.0;
        let q2 = self.q.1;
        let q3 = self.q.2;
        let q4 = self.q.3;

        let q1_q2 = q1 * q2;
        let q1_q3 = q1 * q3;
        let q1_q4 = q1 * q4;

        let q2_q2 = q2 * q2;
        let q2_q3 = q2 * q3;
        let q2_q4 = q2 * q4;

        let q3_q3 = q3 * q3;
        let q3_q4 = q3 * q4;

        let q4_q4 = q4 * q4;

        // f_g: 3x1 matrix (Eq. 25)
        let f_g = &mat!(f32, [
            [2. * (q2_q4 - q1_q3) - a.x],
            [2. * (q1_q2 + q3_q4) - a.y],
            [2. * (0.5 - q2_q2 - q3_q3) - a.z],
        ]);

        // J_g: 3x4 matrix (Eq. 26)
        let j_g = &mat!(f32, [
            [-2. * q3, 2. * q4, -2. * q1, 2. * q2],
            [2. * q2, 2. * q1, 2. * q4, 2. * q3],
            [0., -4. * q2, -4. * q3, 0.],
        ]);

        // f_b: 3x1 matrix (Eq. 29)
        let f_b = &mat!(f32, [
            [2. * bx * (0.5 - q3_q3 - q4_q4) + 2. * bz * (q2_q4 - q1_q3) - m.x],
            [2. * bx * (q2_q3 - q1_q4) + 2. * bz * (q1_q2 + q3_q4) - m.y],
            [2. * bx * (q1_q3 + q2_q4) + 2. * bz * (0.5 - q2_q2 - q3_q3) - m.z],
        ]);

        // J_b: 3x4 matrix (Eq. 30)
        let j_b = &mat!(f32, [
            [
                -2. * bz * q3,
                2. * bz * q4,
                -4. * bx * q3 - 2. * bz * q1,
                -4. * bx * q4 + 2. * bz * q2
            ],
            [
                -2. * bx * q4 + 2. * bz * q2,
                2. * bx * q3 + 2. * bz * q1,
                2. * bx * q2 + 2. * bz * q4,
                -2. * bx * q1 + 2. * bz * q3
            ],
            [
                2. * bx * q3,
                2. * bx * q4 - 4. * bz * q2,
                2. * bx * q1 - 4. * bz * q3,
                2. * bx * q2
            ],
        ]);

        // nabla_f: 4x1 matrix (Eq. 34)
        let nabla_f = j_g.t() * f_g + j_b.t() * f_b;

        // into quaternion
        let mut nabla_f = Quaternion(
            nabla_f.get(0, 0),
            nabla_f.get(1, 0),
            nabla_f.get(2, 0),
            nabla_f.get(3, 0),
        );

        // normalize (beware of division by zero!)
        if nabla_f != Quaternion(0., 0., 0., 0.) {
            nabla_f *= rsqrt(nabla_f.norm());

            // update dqqt (Eq. 43)
            dqdt -= self.beta * nabla_f;
        }

        // update the quaternion (Eq. 42)
        self.q += dqdt * self.sample_period;

        // normalize the quaternion
        self.q *= rsqrt(self.q.norm());

        self.q
    }
}

/// Vector in 3D space
#[derive(Clone, Copy, Debug)]
pub struct F32x3 {
    /// X component
    pub x: f32,
    /// Y component
    pub y: f32,
    /// Z component
    pub z: f32,
}

impl F32x3 {
    /// Returns the norm of this vector
    pub fn norm(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}

impl ops::MulAssign<f32> for F32x3 {
    fn mul_assign(&mut self, k: f32) {
        self.x *= k;
        self.y *= k;
        self.z *= k;
    }
}

/// Quaternion
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion(pub f32, pub f32, pub f32, pub f32);

impl Quaternion {
    /// Returns the conjugate of this quaternion
    pub fn conj(self) -> Self {
        Quaternion(self.0, -self.1, -self.2, -self.3)
    }

    /// Returns the norm of this quaternion
    pub fn norm(self) -> f32 {
        self.0 * self.0 + self.1 * self.1 + self.2 * self.2 + self.3 * self.3
    }
}

fn rsqrt(x: f32) -> f32 {
    1. / x.sqrt()
}

// Fast inverse square root
// XXX what's the effect of the rounding error on the filter?
#[cfg(fast)]
fn rsqrt(x: f32) -> f32 {
    use core::mem;

    let y: f32 = unsafe {
        let i: i32 = mem::transmute(x);
        mem::transmute(0x5f3759df - (i >> 1))
    };

    y * (1.5 - 0.5 * x * y * y)
}

impl ops::AddAssign<Quaternion> for Quaternion {
    fn add_assign(&mut self, rhs: Quaternion) {
        self.0 += rhs.0;
        self.1 += rhs.1;
        self.2 += rhs.2;
        self.3 += rhs.3;
    }
}

impl ops::MulAssign<f32> for Quaternion {
    fn mul_assign(&mut self, k: f32) {
        self.0 *= k;
        self.1 *= k;
        self.2 *= k;
        self.3 *= k;
    }
}

impl ops::SubAssign<Quaternion> for Quaternion {
    fn sub_assign(&mut self, rhs: Quaternion) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
        self.2 -= rhs.2;
        self.3 -= rhs.3;
    }
}

impl ops::Mul<Quaternion> for Quaternion {
    type Output = Self;

    fn mul(self, b: Self) -> Self {
        let a = self;
        Quaternion(
            a.0 * b.0 - a.1 * b.1 - a.2 * b.2 - a.3 * b.3,
            a.0 * b.1 + a.1 * b.0 + a.2 * b.3 - a.3 * b.2,
            a.0 * b.2 - a.1 * b.3 + a.2 * b.0 + a.3 * b.1,
            a.0 * b.3 + a.1 * b.2 - a.2 * b.1 + a.3 * b.0,
        )
    }
}

impl ops::Mul<f32> for Quaternion {
    type Output = Self;

    fn mul(self, k: f32) -> Self {
        Quaternion(self.0 * k, self.1 * k, self.2 * k, self.3 * k)
    }
}

impl ops::Mul<Quaternion> for f32 {
    type Output = Quaternion;

    fn mul(self, q: Quaternion) -> Quaternion {
        q * self
    }
}
