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
#![cfg_attr(rustfmt, rustfmt_skip)]

extern crate libm;
extern crate nalgebra as na;

// use core::ops;
pub use na::{Quaternion, Vector3};
use na::Matrix3x4;

use libm::F32Ext;

/// MARG orientation filter
pub struct Marg {
    beta: f32,
    q: Quaternion<f32>,
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
            q: Quaternion::new(0.0, 0.0, 0.0, 1.0),
            sample_period,
        }
    }

    /// Updates the IMU filter and returns the current estimate
    /// - `g`, gyroscope readings
    /// - `a`, accelerometer readings
    pub fn update_imu(&mut self, g: Vector3<f32>, a: Vector3<f32>) -> Quaternion<f32> {
        // Rate of change of quaternion from gyroscope
        // XXX: find nalgebra operation
        let mut q_dot = Quaternion::new(
            0.5 * (-self.q.i * g.x - self.q.j * g.y - self.q.k * g.z),
            0.5 * (self.q.w * g.x + self.q.j * g.z - self.q.k * g.y),
            0.5 * (self.q.w * g.y - self.q.i * g.z + self.q.k * g.x),
            0.5 * (self.q.w * g.z + self.q.i * g.y - self.q.j * g.x),
        );

        let mut a = a;
        if !((0.0 == a.x) && (0.0 == a.y) && (0.0 == a.z)) {
            // Normalise accelerometer measurement
            a *= rsqrt(a.norm());
            let q2 = self.q * 2.0;

            let q0_4 = 4.0 * self.q.w;
            let q1_4 = 4.0 * self.q.i;
            let q2_4 = 4.0 * self.q.j;
            let q1_8 = 8.0 * self.q.i;
            let q2_8 = 8.0 * self.q.j;

            let q0q0 = self.q.w * self.q.w;
            let q1q1 = self.q.i * self.q.i;
            let q2q2 = self.q.j * self.q.j;
            let q3q3 = self.q.k * self.q.k;

            let mut s0 = q0_4 * q2q2 + q2.j * a.x + q0_4 * q1q1 - q2.i * a.y;
            let mut s1 = q1_4 * q3q3 - q2.k * a.x + 4.0 * q0q0 * self.q.i - q2.w * a.y - q1_4
                + q1_8 * q1q1
                + q1_8 * q2q2
                + q1_4 * a.z;
            let mut s2 = 4.0 * q0q0 * self.q.j + q2.w * a.x + q2_4 * q3q3 - q2.k * a.y - q2_4
                + q2_8 * q1q1
                + q2_8 * q2q2
                + q2_4 * a.z;
            let mut s3 = 4.0 * q1q1 * self.q.k - q2.i * a.x + 4.0 * q2q2 * self.q.k - q2.j * a.y;

            let recip_norm = rsqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
            s0 *= recip_norm;
            s1 *= recip_norm;
            s2 *= recip_norm;
            s3 *= recip_norm;

            q_dot.w -= self.beta * s0;
            q_dot.i -= self.beta * s1;
            q_dot.j -= self.beta * s2;
            q_dot.k -= self.beta * s3;
        }

        self.q += q_dot * self.sample_period;

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
    pub fn update(&mut self, mut m: Vector3<f32>, ar: Vector3<f32>, g: Vector3<f32>) -> Quaternion<f32> {
        // the report calls the accelerometer reading `a`; let's follow suit
        let mut a = g;

        // vector of angular rates
        let omega = Quaternion::new(0., ar.x, ar.y, ar.z);

        // rate of change of quaternion from gyroscope (Eq 11)
        let mut dqdt = 0.5 * self.q * omega;

        // normalize orientation vectors
        a *= rsqrt(a.norm());
        m *= rsqrt(m.norm());

        // direction of the earth's magnetic field (Eq. 45 & 46)
        let h = self.q * Quaternion::new(0., m.x, m.y, m.z) * self.q.conjugate();
        let bx = (h.i * h.i + h.j * h.j).sqrt();
        let bz = h.k;

        // gradient descent
        let q1 = self.q.w;
        let q2 = self.q.i;
        let q3 = self.q.j;
        let q4 = self.q.k;

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
        let f_g = Vector3::new(
            2. * (q2_q4 - q1_q3) - a.x,
            2. * (q1_q2 + q3_q4) - a.y,
            2. * (0.5 - q2_q2 - q3_q3) - a.z,
        );

        // J_g: 3x4 matrix (Eq. 26)
        let j_g = Matrix3x4::new(
            -2. * q3, 2. * q4, -2. * q1, 2. * q2,
            2. * q2, 2. * q1, 2. * q4, 2. * q3,
            0., -4. * q2, -4. * q3, 0.,
        );

        // f_b: 3x1 matrix (Eq. 29)
        let f_b = Vector3::new(
            2. * bx * (0.5 - q3_q3 - q4_q4) + 2. * bz * (q2_q4 - q1_q3) - m.x,
            2. * bx * (q2_q3 - q1_q4) + 2. * bz * (q1_q2 + q3_q4) - m.y,
            2. * bx * (q1_q3 + q2_q4) + 2. * bz * (0.5 - q2_q2 - q3_q3) - m.z
        );

        // J_b: 3x4 matrix (Eq. 30)
        let j_b = Matrix3x4::new(
            -2. * bz * q3, 2. * bz * q4, -4. * bx * q3 - 2. * bz * q1, -4. * bx * q4 + 2. * bz * q2,
            -2. * bx * q4 + 2. * bz * q2, 2. * bx * q3 + 2. * bz * q1, 2. * bx * q2 + 2. * bz * q4, -2. * bx * q1 + 2. * bz * q3,
            2. * bx * q3, 2. * bx * q4 - 4. * bz * q2, 2. * bx * q1 - 4. * bz * q3, 2. * bx * q2
        );

        // nabla_f: 4x1 matrix (Eq. 34)
        let nabla_f = j_g.transpose() * f_g + j_b.transpose() * f_b;
        // into quaternion
        // XXX: is it ok?
        let mut nabla_f = Quaternion::from_vector(nabla_f);

        // normalize (beware of division by zero!)
        if nabla_f != Quaternion::new(0., 0., 0., 0.) {
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
