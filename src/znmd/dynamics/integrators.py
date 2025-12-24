"""
Integrator helper mixin for Zhu–Nakamura dynamics.
"""

import os

import numpy as np

__all__ = ["IntegratorMixin"]


class IntegratorMixin:
    """Provide Velocity-Verlet style propagation helpers."""

    def cal_X(self, X, P, G, time_step):
        au_time_step = time_step / self.unit.au_to_fs
        X_next_step = np.zeros((self.atom_n, 3))
        masses = self.mdif.atom_mass()
        for i in range(self.atom_n):
            atom_m = masses[i]
            X_next_step[i] = (
                X[i]
                + (P[i] / atom_m) * au_time_step
                - (0.5 * G[i] / atom_m) * (au_time_step**2)
            )
        return X_next_step

    def cal_P(self, G_old, G_current, P_old, time_step):
        au_time_step = time_step / self.unit.au_to_fs
        P_current = np.zeros((self.atom_n, 3))
        for i in range(self.atom_n):
            P_current[i] = P_old[i] - 0.5 * (G_old[i] + G_current[i]) * au_time_step
        return P_current

    def cal_E_Kin(self, P):
        kin_energy = np.zeros((self.atom_n, 3))
        masses = self.mdif.atom_mass()
        for i in range(self.atom_n):
            atom_m = masses[i]
            kin_energy[i] = np.power(P[i], 2) / (2 * atom_m)
        return np.sum(kin_energy)

    def back_cal(self, input_file, output_file, old_input_file, X, P, G):
        time_step = self.time_step
        prefix = input_file.split(".")[0]
        for _ in range(int(self.time_step / 0.1)):
            self.mdif.remove_file(prefix)
            time_step = time_step - 0.1
            X = self.cal_X(X, P, G, time_step)
            self.mdif.replace_coordinate(old_input_file, input_file, X)
            self.mdif.run(input_file)
            check_result = self.mdif.check_out(output_file)
            if time_step > 0.1:
                if check_result is not None:
                    print("DENSITY OR ENERGY CONVERGED")
                    print(
                        "Back {0:.2f}(fs) recalculation is convergence.".format(
                            (self.time_step - time_step), ">.2f"
                        )
                    )
                    return X, time_step
                else:
                    print(
                        "Back {0:.2f}(fs) recalculation is misconvergence.".format(
                            (self.time_step - time_step), ">.2f"
                        )
                    )
                    print("Continue the back recalculation")
                    continue
        else:
            print("Back recalculation failed!")
            print("Program exit!")
            os._exit(0)
