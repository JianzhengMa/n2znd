import os
import time
import random
import numpy as np

from ..utils.constants import Unit
from ..utils import HistoryWindow
from .hop_flow import process_double_hop, process_single_hop
from .history_ops import reset_windows, append_windows


class MLWorkflowMixin:
    def _determine_post_hop_state(
        self,
        hop_style: str,
        judge_label: str,
        state_u: int,
        state_d: int,
        current_state: int,
        velocity_factor: float,
    ) -> int:
        if velocity_factor == -1:
            return current_state
        if hop_style == "U-D":
            return state_d
        if hop_style == "D-U":
            return state_u
        if hop_style == "Doub_Hop":
            if judge_label == "current_to_lower":
                return state_d
            if judge_label == "current_to_upper":
                return state_u
        return current_state

    def _log_hop_event(
        self,
        hop_style: str,
        judge_label: str,
        loop: int,
        current_time: float,
        state_u: int,
        state_d: int,
        state_c: int,
        velocity_factor: float,
        coordinate_matrix,
    ) -> None:
        if velocity_factor == -1:
            print("Forbidden transitions, no hopping!")
            return

        if hop_style == "U-D":
            src, dst = state_u, state_d
        elif hop_style == "D-U":
            src, dst = state_d, state_u
        else:
            src = state_c
            dst = state_d if judge_label == "current_to_lower" else state_u

        print(f"step:{loop} time:{current_time:.2f} hopping event:{src} -> {dst}")
        self.write_matrix("ConIs.xyz", coordinate_matrix * self.unit.au_to_ang, loop, current_time)
        with open("hopping_flag.log", "a+", encoding="utf-8") as flag_fh:
            flag_fh.write(f"step:{loop} time:{current_time:.1f} hopping event:{src} -> {dst}\n")

    def ml_znmd(self):
        from ase.io import read
        start_time = time.time()
        print("Start Machine Learning Nonadiabatic Molecular Dynamics Simulation")
        print("Localtime:",time.asctime(time.localtime(time.time())))
        print("Current work dictory is:%s\n"%(os.getcwd()))
        print("Machine Learning Nonadiabatic Molecular Dynamics Simulation Method: %s"%self.hop_method)
        print("Check initial input file:")
        self.mdif.check_in(self.init_input)
        print("atom symbol:%s\n"%(set(self.mdif.atom_symbol())))
        print("The electronic calculation program is %s.\n"%self.program_name)
        print("Start check atomic number!")
        self.mdif.check_atom_n()
        self.mdif.check_num_net()
        if os.path.exists(self.init_momentum):
            print("Start check the momentum input file!")
            self.mdif.check_momentum_file()
        print("The fix temperature ensemble is %s\n"%(self.ensemble_type))

        if int(self.state_n) == 1:
            print("Error: The state_n must larger than 1 !")
            exit(0)

        self.mdif.clear_file()
        q_list = HistoryWindow(3)
        g_list = HistoryWindow(3)
        p_list = HistoryWindow(3)
        steps_energy_dict = HistoryWindow(3)

        loop = 0
        time_step = self.time_step
        current_time  = float(-self.time_step)
        nsw = int(self.total_time/self.time_step)

        while loop < nsw:

            print("\n")
            print('{0:-^63}'.format("-"))
            print("-----------------------Output: step = {0}--------------------------".format(loop))
            print('{0:-^63}\n'.format("-"))

            if loop == 0:
                atoms = read(self.init_input,format = "xyz")
                current_state = self.initial_state
                Q = atoms.get_positions() / Unit.au_to_ang
                energy_dict,grad_dict = self.mdif.run(atoms)
                G = grad_dict[str(current_state)]
                if os.path.exists(self.init_momentum):
                    P = self.mdif.ini_momentum(self.init_momentum)
                else:
                    P = self.te.maxwell_boltzmann_momentum(self.initial_temperature)
                Temperature = self.te.get_temperature(P)
            elif loop == 1:
                current_state = self.initial_state
                G_sub_1 = g_list[-1][str(current_state)]
                Q = self.cal_X(q_list[-1],p_list[-1],G_sub_1,time_step)
                atoms.set_positions(Q * Unit.au_to_ang)
                energy_dict,grad_dict = self.mdif.run(atoms)
                G = grad_dict[str(current_state)]
                P_ensemble = self.te.ensemble(momentum=p_list[-1], loop=loop, nsw=nsw)
                P = self.cal_P(G_sub_1,G,P_ensemble,time_step)
                Temperature = self.te.get_temperature(P)
            elif loop >= 2:
                G_sub_1 = g_list[-1][str(current_state)]
                Q = self.cal_X(q_list[-1],p_list[-1],G_sub_1,time_step)
                atoms.set_positions(Q * Unit.au_to_ang)
                energy_dict,grad_dict = self.mdif.run(atoms)
                G = grad_dict[str(current_state)]
                P_ensemble = self.te.ensemble(momentum=p_list[-1], loop=loop, nsw=nsw)
                P = self.cal_P(G_sub_1,G,P_ensemble,time_step)
                Temperature = self.te.get_temperature(P)
            
            sub_1_time = current_time
            current_time = current_time + time_step
            q_list.append(Q)
            g_list.append(grad_dict)
            p_list.append(P)
            steps_energy_dict.append(energy_dict)

            self.mdif.print_matrix(Q*self.unit.au_to_ang,'Coordinate',str(loop),current_time)
            self.mdif.print_matrix(G,'Gradient',str(loop),current_time)
            self.mdif.print_matrix(P,'Momentun',str(loop),current_time)

            kin_energy = self.cal_E_Kin(P)
            energy_list = list(energy_dict.values())
            energy_current_state = energy_dict[str(current_state)] 
            E_tot = energy_current_state + kin_energy
            
            print('\n{0:*^63}\n'.format(" Summary information "))
            print("Step:{0}  time:{1:.2f}(fs)".format(loop,current_time))
            print("Current electronic state:{0}".format(current_state))
            print("Static electronic structure calculation total energy(a.u.):{0}".format(energy_current_state))
            print("Kinetic energy(eV):{0}".format(kin_energy * Unit.au_to_ev))
            print("Temperature: {0} K".format(Temperature))
            print("Total energy (a.u.):{0}".format(E_tot))

            if loop >= 2:
                print("\nCheck the hopping style of previous step between two state!")
                print("Check step {0} ; time {1}(fs) !".format((loop - 1),current_time-time_step))
                
                q_loop_sub_2, q_loop_sub_1, q_loop_current = q_list.window()
                g_loop_sub_2_dict, g_loop_sub_1_dict, g_loop_current_dict = g_list.window()
                p_loop_sub_2, p_loop_sub_1, p_loop_current = p_list.window()
                energy_dict_sub_2, energy_dict_sub_1, energy_dict_current = steps_energy_dict.window()
                
                h_s,U_u,U_d,state_u,state_d = self.check_single_hop_style(\
                                                                current_state,energy_dict_sub_2,\
                                                                energy_dict_sub_1,energy_dict_current)

                g_loop_sub_2 = g_loop_sub_2_dict[str(current_state)]
                g_loop_sub_1 = g_loop_sub_1_dict[str(current_state)]
                g_loop_current = g_loop_current_dict[str(current_state)]

                if h_s == None:
                    h_s,U_c,U_d,U_u,state_c,state_d,state_u \
                                                        = self.check_double_hop_style(current_state,energy_dict_sub_2,\
                                                                                        energy_dict_sub_1,energy_dict_current)
                    if h_s == None:
                        print('\n{0:*^63}\n'.format("No hopping event"))
                    else:
                        print('\n{0:*^63}\n'.format("Hopping event!!!"))
                        print("Hopping style = {0}".format(h_s))
                        print("Current state: {0}; ".format(state_c),"Current energy level: {0}".format(U_c))
                        print("Upper state: {0}".format(state_u),"Upper energy level: {0}".format(U_u))
                        print("Lower state: {0}; ".format(state_d),"Lower energy level: {0}".format(U_d))
                else:
                    print('\n{0:*^63}\n'.format("Hopping event!!!"))
                    print("Hopping style = {0}".format(h_s))
                    print("Upper state: {0}; ".format(state_u),"Upper energy level: {0}".format(U_u))
                    print("Lower state: {0}; ".format(state_d),"Lower energy level: {0}".format(U_d))

                if h_s is not None:
                    q_window = (q_loop_sub_2, q_loop_sub_1, q_loop_current)
                    base_gradients = (g_loop_sub_2, g_loop_sub_1, g_loop_current)

                    judge_label = None
                    if h_s in ("U-D", "D-U"):
                        judge_label = "current_to_lower" if h_s == "U-D" else "current_to_upper"

                        def fetch_ml_gradients(direction):
                            target_state = state_d if direction == "U-D" else state_u
                            gradients = (
                                g_loop_sub_2_dict[str(target_state)],
                                g_loop_sub_1_dict[str(target_state)],
                                g_loop_current_dict[str(target_state)],
                            )
                            return gradients, None

                        single_result = process_single_hop(
                            self,
                            h_s,
                            fetch_ml_gradients,
                            base_gradients,
                            q_window,
                            p_loop_sub_1,
                            (U_u, U_d),
                            begin_title=" Print single adjacent gradient ",
                            end_title=" End of gradient print ",
                        )
                        hop_outcome = single_result.outcome
                        hopping_p = hop_outcome.probability
                        P_new_sub_1 = hop_outcome.momentum
                        v_factor_sub_1 = hop_outcome.velocity_factor
                    elif h_s == "Doub_Hop":

                        def fetch_double_gradients():
                            grads_up = (
                                g_loop_sub_2_dict[str(state_u)],
                                g_loop_sub_1_dict[str(state_u)],
                                g_loop_current_dict[str(state_u)],
                            )
                            grads_down = (
                                g_loop_sub_2_dict[str(state_d)],
                                g_loop_sub_1_dict[str(state_d)],
                                g_loop_current_dict[str(state_d)],
                            )
                            return grads_up, None, grads_down, None

                        double_result = process_double_hop(
                            self,
                            fetch_double_gradients,
                            q_window,
                            p_loop_sub_1,
                            energies_ud=(U_c, U_d),
                            energies_du=(U_u, U_c),
                            base_gradients=base_gradients,
                            begin_title=" Print double adjacent gradient ",
                            end_title=" End of gradient print ",
                        )

                        lower_outcome = double_result.to_lower
                        upper_outcome = double_result.to_upper
                        (
                            g_loop_sub_2_up,
                            g_loop_sub_1_up,
                            g_loop_current_up,
                        ) = double_result.gradients_up
                        (
                            g_loop_sub_2_down,
                            g_loop_sub_1_down,
                            g_loop_current_down,
                        ) = double_result.gradients_down

                        print("The hopping probability of current state to lower state: {0}".format(lower_outcome.probability))

                        print("The hopping probability of current state to upper state: {0}".format(upper_outcome.probability))

                        if lower_outcome.probability >= upper_outcome.probability:
                            hopping_p = lower_outcome.probability
                            P_new_sub_1 = lower_outcome.momentum
                            v_factor_sub_1 = lower_outcome.velocity_factor
                            judge_label = "current_to_lower"
                            print("The hopping probability of current state to lower state is larger than to upper state")
                        else:
                            hopping_p = upper_outcome.probability
                            P_new_sub_1 = upper_outcome.momentum
                            v_factor_sub_1 = upper_outcome.velocity_factor
                            judge_label = "current_to_upper"
                            print("The hopping probability of current state to lower state is smaller than to upper state")

                    random_number = random.random()
                    print("Hopping probability: {0}\tRandom_number: {1}".format(hopping_p,random_number))

                    if hopping_p >= random_number:
                        print('{0:*^63}'.format(" Recalculate the current steps! "))
                        if v_factor_sub_1 == -1:
                            print('\n{0:*^63}\n'.format(" Forbidden transitions! "))
                        else:
                            print('\n{0:*^63}\n'.format(" Hopping successful! "))

                        current_state = self._determine_post_hop_state(
                            h_s, judge_label, state_u, state_d, current_state, v_factor_sub_1
                        )

                        G_sub_1 = g_loop_sub_1_dict[str(current_state)]
                        Q_sub_1 = q_loop_sub_1
                        P_sub_1 = P_new_sub_1

                        reset_windows(
                            (None, q_list, g_list, p_list, steps_energy_dict),
                            (None, Q_sub_1, g_loop_sub_1_dict, P_sub_1, energy_dict_sub_1),
                        )

                        Q = self.cal_X(Q_sub_1,P_sub_1,G_sub_1,time_step)
                        atoms.set_positions(Q * Unit.au_to_ang)
                        energy_dict,grad_dict = self.mdif.run(atoms)
                        G = grad_dict[str(current_state)]
                        P_ensemble = self.te.ensemble(momentum=P_sub_1, loop=loop, nsw=nsw)
                        P = self.cal_P(G_sub_1,G,P_ensemble,time_step)
                        Temperature = self.te.get_temperature(P)
                            
                        current_time = sub_1_time + time_step
                        append_windows(
                            (q_list, g_list, p_list, steps_energy_dict),
                            (Q, grad_dict, P, energy_dict),
                        )

                        self.mdif.print_matrix(Q*self.unit.au_to_ang,'Recalculate Coordinate',str(loop),current_time)
                        self.mdif.print_matrix(G,'Recalculate Gradient',str(loop),current_time)
                        self.mdif.print_matrix(P,'Recalculate Momentun',str(loop),current_time)                          

                        kin_energy = self.cal_E_Kin(P)
                        energy_list = list(energy_dict.values())
                        energy_current_state = energy_dict[str(current_state)] 
                        E_tot = energy_current_state + kin_energy
            
                        print('\n{0:*^63}\n'.format(" Recalculate Summary information "))
                        print("Step:{0}  time:{1:.2f}(fs)".format(loop,current_time))
                        print("Current electronic state:{0}".format(current_state))
                        print("Static electronic structure calculation total energy(a.u.):{0}".format(energy_current_state))
                        print("Kinetic energy(eV):{0}".format(kin_energy * Unit.au_to_ev))
                        print("Temperature: {0} K".format(Temperature))
                        print("Total energy (a.u.):{0}".format(E_tot))

                        self._log_hop_event(
                            h_s,
                            judge_label,
                            loop,
                            current_time,
                            state_u,
                            state_d,
                            state_c if h_s == "Doub_Hop" else current_state,
                            v_factor_sub_1,
                            Q,
                        )
                    else:
                        print("Hopping failure!")

            self.write_matrix("coordinate.xyz",Q*self.unit.au_to_ang,loop,current_time)
            self.write_matrix("grdient.log",G,loop,current_time)
            self.write_matrix("momentum.log",P,loop,current_time)
            self.write_energy("energy.log",[E_tot,energy_current_state,kin_energy] + energy_list,loop,current_time)
            self.write_temperature("temperature.log",Temperature,loop,current_time)
            self.write_state("state.log",int(current_state),loop,current_time)

            loop = loop + 1
        
        end_time = time.time()
        print('\n{0:*^63}'.format(" Time consuming "))
        print("Time consuming:\t{0}(seconds)".format(end_time - start_time))

    def ml_bomd_on_the_fly(self):
        from ase.io import read

        start_time = time.time()
        print("Start Machine Learning Born-Oppenheimer Molecular Dynamics Simulation")
        print("Localtime:", time.asctime(time.localtime(time.time())))
        print("Current work dictory is:%s\n" % (os.getcwd()))
        print("Check initial input file:")
        self.mdif.check_in(self.init_input)
        print("atom symbol:%s\n" % (set(self.mdif.atom_symbol())))
        print("The electronic calculation program is %s.\n" % self.program_name)
        self.mdif.check_atom_n()
        self.mdif.check_num_net()
        if os.path.exists(self.init_momentum):
            print("Start check the momentum input file!")
            self.mdif.check_momentum_file()
        print("The fix temperature ensemble is %s\n" % (self.ensemble_type))

        atoms = read(self.init_input, format="xyz")
        state_key = str(self.initial_state)
        loop = 0
        time_step = self.time_step
        current_time = 0.0
        nsw = int(self.total_time / self.time_step)

        self.mdif.clear_file()
        Q_current = atoms.get_positions() / Unit.au_to_ang
        energy_dict_current, grad_dict_current = self.mdif.run(atoms)
        if state_key not in energy_dict_current:
            available = ", ".join(sorted(energy_dict_current.keys()))
            raise ValueError(
                f"initial_state={self.initial_state} is not provided by MLFF nets; "
                f"available states: {available or 'none'}."
            )
        G_current = grad_dict_current[state_key]
        if os.path.exists(self.init_momentum):
            P_current = self.mdif.ini_momentum(self.init_momentum)
        else:
            P_current = self.te.maxwell_boltzmann_momentum(self.initial_temperature)
        temperature = self.te.get_temperature(P_current)

        while loop < nsw:
            print("\n")
            print('{0:-^63}'.format("-"))
            print("-----------------------Output: step = {0}--------------------------".format(loop))
            print('{0:-^63}\n'.format("-"))

            self._log_bomd_step(
                Q_current,
                G_current,
                P_current,
                energy_dict_current,
                state_key,
                temperature,
                loop,
                current_time,
            )

            loop += 1
            if loop >= nsw:
                break

            Q_next = self.cal_X(Q_current, P_current, G_current, time_step)
            atoms.set_positions(Q_next * Unit.au_to_ang)
            energy_dict_next, grad_dict_next = self.mdif.run(atoms)
            G_next = grad_dict_next[state_key]
            P_ensemble = self.te.ensemble(momentum=P_current, loop=loop, nsw=nsw)
            P_next = self.cal_P(G_current, G_next, P_ensemble, time_step)

            Q_current = Q_next
            P_current = P_next
            G_current = G_next
            energy_dict_current = energy_dict_next
            temperature = self.te.get_temperature(P_current)
            current_time = current_time + time_step

        end_time = time.time()
        print('\n{0:*^63}'.format(" Time consuming "))
        print("Time consuming:\t{0}(seconds)".format(end_time - start_time))


