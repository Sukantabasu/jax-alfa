import jax
import jax.numpy as jnp
import numpy as np


# FAULTY VERSION (from original Utilities.py)
@jax.jit
def Roots_Faulty(coeffs, init_guess=1.0, tol=1e-6, max_iter=20):
    """
    Original faulty version from Utilities.py
    """

    def polynomial(p, x):
        return jnp.polyval(p, x)

    def derivative(p, x):
        dp = jnp.polyder(p)
        return jnp.polyval(dp, x)

    def second_derivative(p, x):
        d2p = jnp.polyder(jnp.polyder(p))
        return jnp.polyval(d2p, x)

    def laguerre_step(x, _):
        f_x = polynomial(coeffs, x)
        df_x = derivative(coeffs, x)
        d2f_x = second_derivative(coeffs, x)

        G = df_x / (f_x + 1e-10)  # Avoid division by zero
        H = G ** 2 - d2f_x / (f_x + 1e-10)

        denom1 = G + jnp.sqrt(
            (coeffs.shape[0] - 1) * (coeffs.shape[0] * H - G ** 2))
        denom2 = G - jnp.sqrt(
            (coeffs.shape[0] - 1) * (coeffs.shape[0] * H - G ** 2))

        denom = jnp.where(jnp.abs(denom1) > jnp.abs(denom2), denom1, denom2)

        x_new = x - (coeffs.shape[0] - 1) / (denom + 1e-10)
        return x_new, jnp.abs(x_new - x) < tol

    # FAULTY: This scan usage is wrong
    root, converged = jax.lax.scan(laguerre_step, init_guess, None,
                                   length=max_iter)

    # FAULTY: converged is an array, not a scalar boolean
    return jnp.where(converged, root, jnp.nan)


# CORRECTED VERSION
@jax.jit
def Roots_Corrected(coeffs, init_guess=1.0, tol=1e-6, max_iter=20):
    """
    Corrected version using while_loop
    """

    def polynomial(p, x):
        return jnp.polyval(p, x)

    def derivative(p, x):
        dp = jnp.polyder(p)
        return jnp.polyval(dp, x)

    def second_derivative(p, x):
        d2p = jnp.polyder(jnp.polyder(p))
        return jnp.polyval(d2p, x)

    def cond_fn(state):
        x, iteration, converged = state
        return (iteration < max_iter) & (~converged)

    def body_fn(state):
        x, iteration, converged = state

        f_x = polynomial(coeffs, x)
        df_x = derivative(coeffs, x)
        d2f_x = second_derivative(coeffs, x)

        G = df_x / (f_x + 1e-10)  # Avoid division by zero
        H = G ** 2 - d2f_x / (f_x + 1e-10)

        denom1 = G + jnp.sqrt(
            (coeffs.shape[0] - 1) * (coeffs.shape[0] * H - G ** 2))
        denom2 = G - jnp.sqrt(
            (coeffs.shape[0] - 1) * (coeffs.shape[0] * H - G ** 2))

        denom = jnp.where(jnp.abs(denom1) > jnp.abs(denom2), denom1, denom2)

        x_new = x - (coeffs.shape[0] - 1) / (denom + 1e-10)

        new_converged = jnp.abs(x_new - x) < tol

        return x_new, iteration + 1, new_converged

    # Initial state: (x, iteration, converged)
    init_state = (init_guess, 0, False)
    final_x, final_iter, final_converged = jax.lax.while_loop(cond_fn, body_fn,
                                                              init_state)

    return jnp.where(final_converged, final_x, jnp.nan)


def analyze_faulty_behavior():
    """
    Analyze what the faulty version actually returns
    """
    print("ANALYSIS OF FAULTY VERSION BEHAVIOR")
    print("=" * 50)

    # Simple test case: x^2 - 4 = 0 (roots at ±2)
    coeffs = jnp.array([1.0, 0.0, -4.0])
    init_guess = 1.0

    print(f"Test polynomial: x^2 - 4 = 0")
    print(f"Expected roots: ±2")
    print(f"Initial guess: {init_guess}")

    # Let's manually trace what happens in the faulty scan
    def laguerre_step(x, _):
        f_x = jnp.polyval(coeffs, x)
        df_x = jnp.polyval(jnp.polyder(coeffs), x)
        d2f_x = jnp.polyval(jnp.polyder(jnp.polyder(coeffs)), x)

        G = df_x / (f_x + 1e-10)
        H = G ** 2 - d2f_x / (f_x + 1e-10)

        denom1 = G + jnp.sqrt(
            (coeffs.shape[0] - 1) * (coeffs.shape[0] * H - G ** 2))
        denom2 = G - jnp.sqrt(
            (coeffs.shape[0] - 1) * (coeffs.shape[0] * H - G ** 2))

        denom = jnp.where(jnp.abs(denom1) > jnp.abs(denom2), denom1, denom2)

        x_new = x - (coeffs.shape[0] - 1) / (denom + 1e-10)
        return x_new, jnp.abs(x_new - x) < 1e-6

    # What scan actually returns
    final_x, convergence_array = jax.lax.scan(laguerre_step, init_guess, None,
                                              length=5)

    print(f"\nWhat scan returns:")
    print(f"final_x: {final_x}")
    print(f"convergence_array: {convergence_array}")
    print(f"convergence_array shape: {convergence_array.shape}")
    print(f"convergence_array dtype: {convergence_array.dtype}")

    # What the faulty code tries to do
    try:
        faulty_result = jnp.where(convergence_array, final_x, jnp.nan)
        print(f"\nFaulty jnp.where result: {faulty_result}")
        print(f"Faulty result shape: {faulty_result.shape}")
        print(f"Faulty result dtype: {faulty_result.dtype}")
    except Exception as e:
        print(f"\nFaulty jnp.where error: {e}")


def compare_results():
    """
    Compare results from both versions on various test cases
    """
    print("\nCOMPARISON OF RESULTS")
    print("=" * 50)

    test_cases = [
        ([1.0, 0.0, -4.0], 1.0, "x^2 - 4 = 0"),
        ([1.0, 0.0, 0.0, -1.0], 0.5, "x^3 - 1 = 0"),
        ([1.0, -3.0], 1.0, "x - 3 = 0"),
        ([0.1, 0.3, 0.6, 0.1, 2.0, 1.0], 1.0,
         "0.1*x^5 + 0.3*x^4 + 0.6*x^3 + 0.1*x^2 + 2*x + 1")
    ]

    for coeffs_list, guess, description in test_cases:
        coeffs = jnp.array(coeffs_list)

        print(f"\nTest: {description}")
        print(f"Initial guess: {guess}")

        try:
            faulty_result = Roots_Faulty(coeffs, guess)
            print(f"Faulty version result: {faulty_result}")
            print(f"Faulty result type: {type(faulty_result)}")
            print(
                f"Faulty result shape: {faulty_result.shape if hasattr(faulty_result, 'shape') else 'N/A'}")

            if hasattr(faulty_result, 'shape') and faulty_result.shape == ():
                # It's a scalar, try to verify
                verification_faulty = jnp.polyval(coeffs, faulty_result)
                print(
                    f"Faulty verification: f({faulty_result}) = {verification_faulty}")
            else:
                print("Faulty result is not a scalar - cannot verify")

        except Exception as e:
            print(f"Faulty version ERROR: {e}")

        try:
            correct_result = Roots_Corrected(coeffs, guess)
            print(f"Corrected version result: {correct_result}")

            if not jnp.isnan(correct_result):
                verification_correct = jnp.polyval(coeffs, correct_result)
                print(
                    f"Correct verification: f({correct_result}) = {verification_correct}")
            else:
                print("Corrected version did not converge")

        except Exception as e:
            print(f"Corrected version ERROR: {e}")


if __name__ == "__main__":
    analyze_faulty_behavior()
    compare_results()