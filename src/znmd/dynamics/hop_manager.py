"""
Hopping decision helpers extracted from the Zhu–Nakamura driver.
"""

import numpy as np

__all__ = ["HopMixin"]


class HopMixin:
    def cal_hop_factor(
        self,
        h_s,
        q1,
        q2,
        q3,
        P,
        U_u,
        U_d,
        g1_u,
        g2_u,
        g3_u,
        g1_d,
        g2_d,
        g3_d,
    ):
        displacement = q3 - q1
        # Only treat geometries as identical when every Cartesian component stays within tolerance.
        if np.max(np.abs(displacement)) <= 0.00005:
            F1 = np.zeros((self.atom_n, 3))
            F2 = np.zeros((self.atom_n, 3))
        else:
            F1 = np.zeros((self.atom_n, 3))
            F2 = np.zeros((self.atom_n, 3))
            if h_s == "U-D":
                for i in range(self.atom_n):
                    F1[i] = -1 * ((g3_d[i] * (q2[i] - q1[i]) - g1_u[i] * (q2[i] - q3[i])) / (q3[i] - q1[i]))
                    F2[i] = -1 * ((g3_u[i] * (q2[i] - q1[i]) - g1_d[i] * (q2[i] - q3[i])) / (q3[i] - q1[i]))
            elif h_s == "D-U":
                for i in range(self.atom_n):
                    F1[i] = -1 * ((g3_u[i] * (q2[i] - q1[i]) - g1_d[i] * (q2[i] - q3[i])) / (q3[i] - q1[i]))
                    F2[i] = -1 * ((g3_d[i] * (q2[i] - q1[i]) - g1_u[i] * (q2[i] - q3[i])) / (q3[i] - q1[i]))

        print("\nCalculate the diabatic force(steps - 1): F1 , F2")
        self.mdif.print_matrix(F1, "F1", "xxx", "xxx")
        self.mdif.print_matrix(F2, "F2", "xxx", "xxx")

        F1_sub_F2_2 = np.zeros((self.atom_n, 3))
        F1_mul_F2 = np.zeros((self.atom_n, 3))
        s_i = np.zeros((self.atom_n, 3))
        for i in range(self.atom_n):
            atom_m = self.mdif.atom_mass()[i]
            F1_sub_F2_2[i] = np.power((F1[i] - F2[i]), 2) / atom_m
            F1_mul_F2[i] = F1[i] * F2[i] / atom_m
            s_i[i] = (F2[i] - F1[i]) / np.sqrt(atom_m)
        sum_F1_sub_F2_2 = np.sum(F1_sub_F2_2)
        sum_F1_mul_F2 = np.sum(F1_mul_F2)

        s_i_norm = np.zeros((self.atom_n, 3))
        n_i = np.zeros((self.atom_n, 3))
        P_parallel = np.zeros((self.atom_n, 3))
        P_vertical = np.zeros((self.atom_n, 3))

        for i in range(self.atom_n):
            s_i_norm[i] = s_i[i] / np.sqrt(sum_F1_sub_F2_2)

            if (s_i_norm[i][0] == 0.0) and (s_i_norm[i][1] == 0.0) and (s_i_norm[i][2] == 0.0):
                s_i_norm[i][0] = 1.0
                s_i_norm[i][1] = 1.0
                s_i_norm[i][2] = 1.0
                n_i[i] = s_i_norm[i] / np.linalg.norm(s_i_norm[i])
            else:
                n_i[i] = s_i_norm[i] / np.linalg.norm(s_i_norm[i])

            P_parallel[i] = np.dot(P[i], n_i[i]) * n_i[i]
            P_vertical[i] = P[i] - P_parallel[i]

        print("\nSplit P in vertical and parallel direction\n")
        self.mdif.print_matrix(n_i, "n_i", "xxx", "xxx")
        self.mdif.print_matrix(P_parallel, "P_parallel", "xxx", "xxx")
        self.mdif.print_matrix(P_vertical, "P_vertical", "xxx", "xxx")

        E_kin_parallel = self.cal_E_Kin(P_parallel)
        E_kin_vertical = self.cal_E_Kin(P_vertical)
        print("\nCalculate kinetic energy in parallel and vertical direction:")
        print(f"E_kin_parallel = {E_kin_parallel}")
        print(f"E_kin_vertical = {E_kin_vertical}")

        if h_s == "U-D":
            E_t = U_u + E_kin_parallel
        elif h_s == "D-U":
            E_t = U_d + E_kin_parallel
        E_x = (U_u + U_d) / 2
        V_12 = np.abs((U_u - U_d)) / 2
        print(f"E_t (Adjusted energy) = {E_t}")
        print(f"E_x (Energy at intersection point) = {E_x}")
        print(f"V_12 (Diabatic coupling) = {V_12}")

        if self.hop_method == "ZN":
            aa = (np.sqrt(sum_F1_sub_F2_2) * np.sqrt(np.abs(sum_F1_mul_F2))) / (16 * V_12**3)
            bb = (E_t - E_x) * np.sqrt(sum_F1_sub_F2_2) / ((2 * V_12) * np.sqrt(np.abs(sum_F1_mul_F2)))
            print("method = ZN")
        elif self.hop_method == "ZN_sim":
            aa = sum_F1_sub_F2_2 / (16 * V_12**3)
            bb = (E_t - E_x) / (2 * V_12)
            print("method = ZN_sim")
        else:
            print("Error: The method variable is invalidity. Please input the right method\n")

        print(f"a*a = {aa}")
        print(f"b*b = {bb}")

        if h_s == "U-D":
            v_factor = np.sqrt(1 + (V_12 * 2) / E_kin_parallel)
        elif h_s == "D-U":
            v_factor_square = 1 - (V_12 * 2) / E_kin_parallel
            if v_factor_square >= 0:
                v_factor = np.sqrt(v_factor_square)
            else:
                v_factor = -1

        new_P_parallel = P_parallel * v_factor
        new_P = new_P_parallel + P_vertical
        self.mdif.print_matrix(new_P, "Adjust P after hopping", "xxx", "xxx")
        print(f"The parallel momentum rescale factor v_factor: {v_factor}")

        if sum_F1_mul_F2 >= 0:
            if aa >= 1000:
                hop_p = 1
            elif aa < 0.001:
                hop_p = 0
            else:
                hop_p = np.exp(
                    (-1 * self.unit.pi / (4 * np.sqrt(aa)))
                    * np.sqrt(2 / (bb + np.sqrt(np.power(bb, 2) + 1)))
                )
                print("The conical intersection is same sign slope.\n")
        else:
            if aa >= 1000:
                hop_p = 1
            elif aa < 0.001:
                hop_p = 0
            else:
                print("The conical intersection is opposite sign slope.\n")
                hop_p = np.exp(
                    (-1 * self.unit.pi / (4 * np.sqrt(aa)))
                    * np.sqrt(2 / (bb + np.sqrt(abs(np.power(bb, 2) - 1))))
                )

        return hop_p, new_P, v_factor

    def check_single_hop_style(self, current_state, energy_dict_sub_2, energy_dict_sub_1, energy_dict_current):
        current_state = int(current_state)
        print("\n{0:*^63}".format(" Check the single hop style "))

        if int(current_state) == 1:
            energy_current_state_sub_2 = energy_dict_sub_2[str(current_state)]
            energy_current_state_sub_1 = energy_dict_sub_1[str(current_state)]
            energy_current_state_current = energy_dict_current[str(current_state)]

            energy_up_state_sub_2 = energy_dict_sub_2[str(int(current_state + 1))]
            energy_up_state_sub_1 = energy_dict_sub_1[str(int(current_state + 1))]
            energy_up_state_current = energy_dict_current[str(int(current_state + 1))]

            enegry_gap_sub_2 = abs(energy_up_state_sub_2 - energy_current_state_sub_2) * self.unit.au_to_ev
            enegry_gap_sub_1 = abs(energy_up_state_sub_1 - energy_current_state_sub_1) * self.unit.au_to_ev
            enegry_gap_current = abs(energy_up_state_current - energy_current_state_current) * self.unit.au_to_ev

            print("Enegry gap between current state and first excited state (ev):")
            print(
                "current steps-2: {0}\tcurrent steps-1: {1}\tcurrent steps: {2}".format(
                    enegry_gap_sub_2, enegry_gap_sub_1, enegry_gap_current
                )
            )
            if (
                enegry_gap_sub_2 > enegry_gap_sub_1 < enegry_gap_current
                and enegry_gap_sub_1 < self.hopping_threshold_value
            ):
                U_u = energy_up_state_sub_1
                U_d = energy_current_state_sub_1
                state_u = current_state + 1
                state_d = current_state
                return "D-U", U_u, U_d, state_u, state_d
            else:
                return None, None, None, None, None

        elif 1 < int(current_state) < self.state_n:
            energy_down_state_sub_2 = energy_dict_sub_2[str(int(current_state - 1))]
            energy_down_state_sub_1 = energy_dict_sub_1[str(int(current_state - 1))]
            energy_down_state_current = energy_dict_current[str(int(current_state - 1))]

            energy_current_state_sub_2 = energy_dict_sub_2[str(current_state)]
            energy_current_state_sub_1 = energy_dict_sub_1[str(current_state)]
            energy_current_state_current = energy_dict_current[str(current_state)]

            energy_up_state_sub_2 = energy_dict_sub_2[str(int(current_state + 1))]
            energy_up_state_sub_1 = energy_dict_sub_1[str(int(current_state + 1))]
            energy_up_state_current = energy_dict_current[str(int(current_state + 1))]

            enegry_gap_c_d_state_sub_2 = abs(energy_current_state_sub_2 - energy_down_state_sub_2) * self.unit.au_to_ev
            enegry_gap_c_d_state_sub_1 = abs(energy_current_state_sub_1 - energy_down_state_sub_1) * self.unit.au_to_ev
            enegry_gap_c_d_state_current = (
                abs(energy_current_state_current - energy_down_state_current) * self.unit.au_to_ev
            )

            enegry_gap_c_u_state_sub_2 = abs(energy_up_state_sub_2 - energy_current_state_sub_2) * self.unit.au_to_ev
            enegry_gap_c_u_state_sub_1 = abs(energy_up_state_sub_1 - energy_current_state_sub_1) * self.unit.au_to_ev
            enegry_gap_c_u_state_current = (
                abs(energy_up_state_current - energy_current_state_current) * self.unit.au_to_ev
            )

            print("Enegry gap between current state and upper level state (ev):")
            print(
                "current steps-2: {0}\tcurrent steps-1: {1}\tcurrent steps: {2}".format(
                    enegry_gap_c_u_state_sub_2, enegry_gap_c_u_state_sub_1, enegry_gap_c_u_state_current
                )
            )

            print("Enegry gap between current state and lower level state (ev):")
            print(
                "current steps-2: {0}\tcurrent steps-1: {1}\tcurrent steps: {2}".format(
                    enegry_gap_c_d_state_sub_2, enegry_gap_c_d_state_sub_1, enegry_gap_c_d_state_current
                )
            )

            hop_c_d_judge_criteria = (
                enegry_gap_c_d_state_sub_2 > enegry_gap_c_d_state_sub_1 < enegry_gap_c_d_state_current
                and enegry_gap_c_d_state_sub_1 < self.hopping_threshold_value
            )

            hop_c_u_judge_criteria = (
                enegry_gap_c_u_state_sub_2 > enegry_gap_c_u_state_sub_1 < enegry_gap_c_u_state_current
                and enegry_gap_c_u_state_sub_1 < self.hopping_threshold_value
            )

            if hop_c_d_judge_criteria and (not hop_c_u_judge_criteria):
                U_u = energy_current_state_sub_1
                U_d = energy_down_state_sub_1
                state_u = current_state
                state_d = current_state - 1
                return "U-D", U_u, U_d, state_u, state_d
            elif hop_c_u_judge_criteria and (not hop_c_d_judge_criteria):
                U_u = energy_up_state_sub_1
                U_d = energy_current_state_sub_1
                state_u = current_state + 1
                state_d = current_state
                return "D-U", U_u, U_d, state_u, state_d
            else:
                return None, None, None, None, None

        elif int(current_state) == self.state_n:
            energy_current_state_sub_2 = energy_dict_sub_2[str(current_state)]
            energy_current_state_sub_1 = energy_dict_sub_1[str(current_state)]
            energy_current_state_current = energy_dict_current[str(current_state)]

            energy_down_state_sub_2 = energy_dict_sub_2[str(int(current_state - 1))]
            energy_down_state_sub_1 = energy_dict_sub_1[str(int(current_state - 1))]
            energy_down_state_current = energy_dict_current[str(int(current_state - 1))]

            enegry_gap_sub_2 = abs(energy_current_state_sub_2 - energy_down_state_sub_2) * self.unit.au_to_ev
            enegry_gap_sub_1 = abs(energy_current_state_sub_1 - energy_down_state_sub_1) * self.unit.au_to_ev
            enegry_gap_current = abs(energy_current_state_current - energy_down_state_current) * self.unit.au_to_ev

            print("Enegry gap between current state and lower level state (ev):")
            print(
                "current steps-2: {0}\tcurrent steps-1: {1}\tcurrent steps: {2}".format(
                    enegry_gap_sub_2, enegry_gap_sub_1, enegry_gap_current
                )
            )

            if (
                enegry_gap_sub_2 > enegry_gap_sub_1 < enegry_gap_current
                and enegry_gap_sub_1 < self.hopping_threshold_value
            ):
                U_u = energy_current_state_sub_1
                U_d = energy_down_state_sub_1
                state_u = current_state
                state_d = current_state - 1
                return "U-D", U_u, U_d, state_u, state_d
            else:
                return None, None, None, None, None

    def check_double_hop_style(self, current_state, energy_dict_sub_2, energy_dict_sub_1, energy_dict_current):
        current_state = int(current_state)
        print("\n{0:*^63}".format(" Check the double hop style "))

        if 1 < int(current_state) < self.state_n:
            energy_down_state_sub_2 = energy_dict_sub_2[str(int(current_state - 1))]
            energy_down_state_sub_1 = energy_dict_sub_1[str(int(current_state - 1))]
            energy_down_state_current = energy_dict_current[str(int(current_state - 1))]

            energy_current_state_sub_2 = energy_dict_sub_2[str(current_state)]
            energy_current_state_sub_1 = energy_dict_sub_1[str(current_state)]
            energy_current_state_current = energy_dict_current[str(current_state)]

            energy_up_state_sub_2 = energy_dict_sub_2[str(int(current_state + 1))]
            energy_up_state_sub_1 = energy_dict_sub_1[str(int(current_state + 1))]
            energy_up_state_current = energy_dict_current[str(int(current_state + 1))]

            enegry_gap_c_d_state_sub_2 = abs(energy_current_state_sub_2 - energy_down_state_sub_2) * self.unit.au_to_ev
            enegry_gap_c_d_state_sub_1 = abs(energy_current_state_sub_1 - energy_down_state_sub_1) * self.unit.au_to_ev
            enegry_gap_c_d_state_current = (
                abs(energy_current_state_current - energy_down_state_current) * self.unit.au_to_ev
            )

            enegry_gap_c_u_state_sub_2 = abs(energy_up_state_sub_2 - energy_current_state_sub_2) * self.unit.au_to_ev
            enegry_gap_c_u_state_sub_1 = abs(energy_up_state_sub_1 - energy_current_state_sub_1) * self.unit.au_to_ev
            enegry_gap_c_u_state_current = (
                abs(energy_up_state_current - energy_current_state_current) * self.unit.au_to_ev
            )

            print("Enegry gap between current state and upper level state (ev):")
            print(
                "current steps-2: {0}\tcurrent steps-1: {1}\tcurrent steps: {2}".format(
                    enegry_gap_c_u_state_sub_2, enegry_gap_c_u_state_sub_1, enegry_gap_c_u_state_current
                )
            )

            print("Enegry gap between current state and lower level state (ev):")
            print(
                "current steps-2: {0}\tcurrent steps-1: {1}\tcurrent steps: {2}".format(
                    enegry_gap_c_d_state_sub_2, enegry_gap_c_d_state_sub_1, enegry_gap_c_d_state_current
                )
            )

            hop_c_d_judge_criteria = (
                enegry_gap_c_d_state_sub_2 > enegry_gap_c_d_state_sub_1 < enegry_gap_c_d_state_current
                and enegry_gap_c_d_state_sub_1 < self.hopping_threshold_value
            )
            hop_c_u_judge_criteria = (
                enegry_gap_c_u_state_sub_2 > enegry_gap_c_u_state_sub_1 < enegry_gap_c_u_state_current
                and enegry_gap_c_u_state_sub_1 < self.hopping_threshold_value
            )

            if hop_c_d_judge_criteria and hop_c_u_judge_criteria:
                U_c = energy_current_state_sub_1
                U_d = energy_down_state_sub_1
                U_u = energy_up_state_sub_1
                state_c = current_state
                state_d = current_state - 1
                state_u = current_state + 1
                return "Doub_Hop", U_c, U_d, U_u, state_c, state_d, state_u
            else:
                return None, None, None, None, None, None, None
        else:
            print("No double hop check!")
            return None, None, None, None, None, None, None

    # cal_single_adjacent_gradient and cal_double_adjacent_gradient remain as in original class
