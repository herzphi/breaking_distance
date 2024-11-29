import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Streamlit app
st.title("Car Braking Distance Simulator")

# User inputs for parameters
st.sidebar.header("Simulation Parameters")
m = st.sidebar.number_input("Mass of the car (kg)", value=2000.0, step=100.0)
T = st.sidebar.number_input("Braking torque per wheel (Nm)", value=1.0, step=0.1)
r_w = st.sidebar.number_input("Wheel radius (m)", value=0.3, step=0.01)
I_w = st.sidebar.number_input(
    "Moment of inertia of each wheel (kg m^2)", value=1.0, step=0.1
)
mu_max = st.sidebar.number_input(
    "Maximum friction coefficient (mu_max)", value=0.8, step=0.05
)
C = st.sidebar.number_input("Friction curve shape parameter C", value=1.5, step=0.1)
D = st.sidebar.number_input("Friction curve shape parameter D", value=1.2, step=0.1)
st.sidebar.page_link("https://de.mathworks.com/help/sdl/ref/tireroadinteractionmagicformula.html", label="Tire-Road Interaction")

# Initial conditions
v0 = st.sidebar.number_input(
    "Initial velocity (m/s)", value=0.27778, step=0.01
)  # 1 km/h
omega0 = v0 / r_w

# Gravitational acceleration
g = 9.81


# Function for mu(lambda)
def mu(lambda_):
    return mu_max * np.sin(C * np.arctan(D * lambda_))


# Differential equations
def braking_dynamics(t, y):
    v, omega = y

    if v <= 0:
        return [0, 0]  # Stop simulation when velocity reaches zero

    # Calculate slip ratio
    lambda_ = (v - omega * r_w) / v

    # Friction coefficient
    mu_lambda = mu(lambda_)

    # Translational acceleration
    dv_dt = -mu_lambda * g

    # Rotational acceleration
    domega_dt = -T / I_w

    return [dv_dt, domega_dt]


# Time span and initial conditions
t_span = (0, v0 * 4)  # Simulate for up to 10 seconds
y0 = [v0, omega0]

# Solve the system of equations
sol = solve_ivp(braking_dynamics, t_span, y0, t_eval=np.linspace(0, int(v0 * 4), 1000))

# Extract results
time = sol.t
velocity = sol.y[0]
angular_velocity = sol.y[1]

# Compute braking distance
distance = np.cumsum(velocity * np.diff(time, prepend=0))

# Plot results
fig, ax = plt.subplots(3, 1, figsize=(8, 12))

# Velocity plot
ax[0].plot(time, velocity, label="Velocity (m/s)")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Velocity (m/s)")
ax[0].set_title("Car Velocity vs Time")
ax[0].grid()
ax[0].legend()

# Angular velocity plot
ax[1].plot(time, angular_velocity, label="Angular Velocity (rad/s)", color="orange")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Angular Velocity (rad/s)")
ax[1].set_title("Wheel Angular Velocity vs Time")
ax[1].grid()
ax[1].legend()

# Braking distance plot
ax[2].plot(time, distance, label="Braking Distance (m)", color="green")
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("Distance (m)")
ax[2].set_title("Braking Distance vs Time")
ax[2].grid()
ax[2].legend()

plt.tight_layout()

tab1, tab2 = st.tabs(["Plot", "Math"])
# Display the plots
with tab1:
    st.pyplot(fig)
with tab2:
    st.markdown(
        """
        The braking dynamics are modeled using the following key equations:

        1. **Slip Ratio**:
           \[
           \lambda = \frac{v - \omega r_w}{v}
           \]
           - \( v \): Translational velocity of the car.
           - \( \omega \): Angular velocity of the wheels.
           - \( r_w \): Radius of the wheels.

        2. **Friction Coefficient**:
           \[
           \mu(\lambda) = \mu_{\text{max}} \cdot \sin(C \cdot \arctan(D \cdot \lambda))
           \]
           - \( \mu_{\text{max}} \): Maximum friction coefficient.
           - \( C, D \): Curve shape parameters.

        3. **Translational Deceleration**:
           \[
           \frac{dv}{dt} = -\mu(\lambda) \cdot g
           \]
           - \( g \): Gravitational acceleration.

        4. **Rotational Deceleration**:
           \[
           \frac{d\omega}{dt} = -\frac{T}{I_w}
           \]
           - \( T \): Braking torque per wheel.
           - \( I_w \): Moment of inertia of each wheel.

        5. **Braking Distance**:
           The braking distance is computed as:
           \[
           d = \int_0^{t_{\text{stop}}} v(t) \, dt
           \]
           - \( t_{\text{stop}} \): Time when the car comes to rest.

        Using Runge Kutta to solve the non linear differential equations.
        """
    )


# Display final braking distance
st.subheader("Results")
st.write(f"Final braking distance: {distance[-1]:.2f} meters")
st.write(f"Time to stop: {time[np.argmax(velocity <= 0)]:.2f} seconds")
