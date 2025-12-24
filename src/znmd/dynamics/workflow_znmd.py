import os
import time
import random
import numpy as np
from pathlib import Path
import shutil


from ..utils import HistoryWindow
from ..utils.constants import Unit
from .hop_flow import process_double_hop, process_single_hop
from .history_ops import reset_windows, append_windows


class ZNMDWorkflowMixin:
    def znmd_on_the_fly(self):
        start_time = time.time()
        print("Start Nonadiabatic Molecular Dynamics Simulation")
        print("Localtime:",time.asctime(time.localtime(time.time())))
        print("Current work dictory is:%s\n"%(os.getcwd()))
        print("Nonadiabatic Molecular Dynamics Simulation Method: %s"%self.hop_method)
        print("Check initial input file:")
        self.mdif.check_in(self.init_input)
        print("atom symbol:%s\n"%(set(self.mdif.atom_symbol())))
        print("The electronic calculation program is %s.\n"%self.program_name)
        print("Start check state number!")
        self.mdif.check_state()
        print("Start check atomic number!")
        self.mdif.check_atom_n()
        if os.path.exists(self.init_momentum):
            print("Start check the momentum input file!")
            self.mdif.check_momentum_file()
        print("The fix temperature ensemble is %s\n"%(self.ensemble_type))
        
        if int(self.state_n) == 1:
            print("Error: The state_n must larger than 1 !")
            exit(0)

        self.mdif.clear_file()
        time_step_list = HistoryWindow(3)
        q_list = HistoryWindow(3)
        g_list = HistoryWindow(3)
        p_list = HistoryWindow(3)
        input_filename_list = HistoryWindow(3)
        output_filename_list = HistoryWindow(3)
        steps_energy_dict= HistoryWindow(3)

        loop = 0
        time_step = self.time_step
        current_time  = float(-self.time_step)
        nsw = int(self.total_time/self.time_step)

        while loop < nsw:
            
            if loop == 0:
                Q = self.mdif.get_coordinate(self.init_input)
                input_file = self.init_input
                output_file = self.init_input.split('.')[0] + ".out"
            elif loop == 1:
                Q = self.cal_X(q_list[-1],p_list[-1],g_list[-1],time_step)
                prefix = self.init_input.split(".")[0]+"_"+str(1)
                input_file,output_file = self.mdif.gen_filename(prefix)
                self.mdif.replace_coordinate(self.init_input,input_file,Q)
            elif loop >= 2:
                Q = self.cal_X(q_list[-1],p_list[-1],g_list[-1],time_step)
                prefix = self.init_input.split(".")[0]+"_"+str(loop)
                input_file,output_file = self.mdif.gen_filename(prefix)
                self.mdif.replace_coordinate(input_filename_list[-1],input_file,Q)
                
            self.mdif.run(input_file)
            check_result = self.mdif.check_out(output_file)
            print('{0:-^63}'.format("-"))
            print("-----------------------Output: step = {0}--------------------------".format(loop))
            print('{0:-^63}\n'.format("-"))

            if check_result != None:
                print("DENSITY OR ENERGY CONVERGED")
                print("Ab initial calculation is convergence!")
                G = self.mdif.get_gradient(output_file)
                if loop == 0:
                    if os.path.exists(self.init_momentum):
                        P = self.mdif.ini_momentum(self.init_momentum)
                    else:
                        P = self.te.maxwell_boltzmann_momentum(self.initial_temperature)
                    Temperature = self.te.get_temperature(P)
                elif loop == 1:
                    P_ensemble = self.te.ensemble(momentum=p_list[-1], loop=loop, nsw=nsw)
                    P = self.cal_P(g_list[-1],G,P_ensemble,time_step)
                    Temperature = self.te.get_temperature(P)
                elif loop >= 2:
                    P_ensemble = self.te.ensemble(momentum=p_list[-1], loop=loop, nsw=nsw)
                    P = self.cal_P(g_list[-1],G,P_ensemble,time_step)
                    Temperature = self.te.get_temperature(P)
                time_step_list.append(time_step)

            elif check_result == None:
                print("Ab initial calculation is misconvergence!")
                if loop == 0:
                    print("Changing the atomic coordinate might solve this problem!")
                    os._exit(0)
                elif loop == 1:
                    print('\n{0:*^63}'.format(" Start Back Recalculation! "))
                    Q,back_time_step = self.back_cal(input_file,output_file,input_filename_list[-1],q_list[-1],p_list[-1],g_list[-1])
                    G = self.mdif.get_gradient(output_file)
                    P_ensemble = self.te.ensemble(momentum=p_list[-1], loop=loop, nsw=nsw)
                    P = self.cal_P(g_list[-1],G,P_ensemble,back_time_step)
                    Temperature = self.te.get_temperature(P)
                elif loop >= 2:
                    print('\n{0:*^63}'.format(" Start Back Recalculation! "))
                    Q,back_time_step = self.back_cal(input_file,output_file,input_filename_list[-1],q_list[-1],p_list[-1],g_list[-1])
                    G = self.mdif.get_gradient(output_file)
                    P_ensemble = self.te.ensemble(momentum=p_list[-1], loop=loop, nsw=nsw)
                    P = self.cal_P(g_list[-1],G,P_ensemble,back_time_step)
                    Temperature = self.te.get_temperature(P)
                time_step = back_time_step
                time_step_list.append(1-time_step)
            
            sub_1_time = current_time
            current_time = current_time + time_step
            q_list.append(Q)
            g_list.append(G)
            p_list.append(P)

            self.mdif.print_matrix(Q*self.unit.au_to_ang,'Coordinate',str(loop),current_time)
            self.mdif.print_matrix(G,'Gradient',str(loop),current_time)
            self.mdif.print_matrix(P,'Momentun',str(loop),current_time)

            kin_energy = self.cal_E_Kin(P)
            current_state = self.mdif.get_state(output_file)
            energy_dict = self.mdif.state_energy(output_file,False)
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

            steps_energy_dict.append(energy_dict)
            input_filename_list.append(input_file)
            output_filename_list.append(output_file)

            if loop >= 2:
                print("\nCheck the hopping style of previous step between two state!")
                print("Check step {0} ; time {1}(fs) filename is: {2}".format((loop - 1),current_time-time_step,output_filename_list[1]))

                time_step_sub_2, time_step_sub_1, time_step_current = time_step_list.window()
                q_loop_sub_2, q_loop_sub_1, q_loop_current = q_list.window()
                g_loop_sub_2, g_loop_sub_1, g_loop_current = g_list.window()
                p_loop_sub_2, p_loop_sub_1, p_loop_current = p_list.window()
                inp_fil_sub_2, inp_fil_sub_1, inp_fil_current = input_filename_list.window()
                out_fil_sub_2, out_fil_sub_1, out_fil_current = output_filename_list.window()
                (
                    energy_dict_sub_2,
                    energy_dict_sub_1,
                    energy_dict_current,
                ) = steps_energy_dict.window()
                
                h_s,U_u,U_d,state_u,state_d = self.check_single_hop_style(\
                                                                current_state,energy_dict_sub_2,\
                                                                energy_dict_sub_1,energy_dict_current)

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

                    if h_s in ("U-D", "D-U"):

                        def fetch_adjacent(direction):
                            gradients = self.cal_single_adjacent_gradient(
                                inp_fil_sub_2,
                                inp_fil_sub_1,
                                inp_fil_current,
                                direction,
                                state_u,
                                state_d,
                            )
                            grad_triplet = gradients[:3]
                            metadata = gradients[3:]
                            return grad_triplet, metadata

                        single_result = process_single_hop(
                            self,
                            h_s,
                            fetch_adjacent,
                            base_gradients,
                            q_window,
                            p_loop_sub_1,
                            (U_u, U_d),
                            begin_title=" Calculate single adjacent gradient ",
                            end_title=" End of gradient calculate ",
                        )
                        hop_outcome = single_result.outcome
                        hopping_p = hop_outcome.probability
                        P_new_sub_1 = hop_outcome.momentum
                        v_factor_sub_1 = hop_outcome.velocity_factor
                        (
                            g_loop_sub_2_new,
                            g_loop_sub_1_new,
                            g_loop_current_new,
                        ) = single_result.gradients
                        (
                            inp_fil_sub_1_temp_new,
                            out_fil_sub_1_temp_new,
                            prefix_sub_1_new,
                        ) = single_result.metadata
                    elif h_s == "Doub_Hop":

                        def fetch_double():
                            gradients = self.cal_double_adjacent_gradient(
                                inp_fil_sub_2,
                                inp_fil_sub_1,
                                inp_fil_current,
                                state_u,
                                state_d,
                            )
                            grads_up = gradients[:3]
                            meta_up = gradients[3:6]
                            grads_down = gradients[6:9]
                            meta_down = gradients[9:]
                            return grads_up, meta_up, grads_down, meta_down

                        double_result = process_double_hop(
                            self,
                            fetch_double,
                            q_window,
                            p_loop_sub_1,
                            energies_ud=(U_c, U_d),
                            energies_du=(U_u, U_c),
                            base_gradients=base_gradients,
                            begin_title=" Calculate double adjacent gradient ",
                            end_title=" End of gradient calculate ",
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
                        (
                            inp_fil_sub_1_temp_up,
                            out_fil_sub_1_temp_up,
                            prefix_sub_1_up,
                        ) = double_result.metadata_up
                        (
                            inp_fil_sub_1_temp_down,
                            out_fil_sub_1_temp_down,
                            prefix_sub_1_down,
                        ) = double_result.metadata_down

                        print("The hopping probability of current state to lower state: {0}".format(lower_outcome.probability))

                        print("The hopping probability of current state to upper state: {0}".format(upper_outcome.probability))

                        if lower_outcome.probability >= upper_outcome.probability:
                            hopping_p = lower_outcome.probability
                            P_new_sub_1 = lower_outcome.momentum
                            v_factor_sub_1 = lower_outcome.velocity_factor
                            self.mdif.remove_file(prefix_sub_1_up)
                            judge_label = "current_to_lower"
                            print("The hopping probability of current state to lower state is larger than to upper state")
                        else:
                            hopping_p = upper_outcome.probability
                            P_new_sub_1 = upper_outcome.momentum
                            v_factor_sub_1 = upper_outcome.velocity_factor
                            self.mdif.remove_file(prefix_sub_1_down)
                            judge_label = "current_to_upper"
                            print("The hopping probability of current state to lower state is smaller than to upper state")

                    random_number = random.random()
                    print("Hopping probability: {0}\tRandom_number: {1}".format(hopping_p,random_number))

                    if hopping_p >= random_number:
                        
                        if v_factor_sub_1 == -1:
                            print('\n{0:*^63}\n'.format(" Forbidden transitions! "))
                        else:
                            print('\n{0:*^63}\n'.format(" Hopping successful! "))

                        print('{0:*^63}'.format(" Recalculate the current steps! "))
                        Path(inp_fil_current).unlink(missing_ok=True)
                        Path(out_fil_current).unlink(missing_ok=True)

                        if h_s in ('U-D', 'D-U'):
                            if v_factor_sub_1 == -1:
                                Path(f"{prefix_sub_1_new}.inp").unlink(missing_ok=True)
                                Path(f"{prefix_sub_1_new}.out").unlink(missing_ok=True)
                                G_sub_1 = g_loop_sub_1
                            else:
                                shutil.move(inp_fil_sub_1_temp_new, inp_fil_sub_1)
                                shutil.move(out_fil_sub_1_temp_new, out_fil_sub_1)
                                G_sub_1 = g_loop_sub_1_new
                        elif h_s == "Doub_Hop":
                            if judge_label == "current_to_lower":
                                if v_factor_sub_1 == -1:
                                    Path(f"{prefix_sub_1_down}.inp").unlink(missing_ok=True)
                                    Path(f"{prefix_sub_1_down}.out").unlink(missing_ok=True)
                                    G_sub_1 = g_loop_sub_1
                                else:
                                    shutil.move(inp_fil_sub_1_temp_down, inp_fil_sub_1)
                                    shutil.move(out_fil_sub_1_temp_down, out_fil_sub_1)
                                    G_sub_1 = g_loop_sub_1_down
                            elif judge_label == "current_to_upper":
                                if v_factor_sub_1 == -1:
                                    Path(f"{prefix_sub_1_up}.inp").unlink(missing_ok=True)
                                    Path(f"{prefix_sub_1_up}.out").unlink(missing_ok=True)
                                    G_sub_1 = g_loop_sub_1
                                else:
                                    shutil.move(inp_fil_sub_1_temp_up, inp_fil_sub_1)
                                    shutil.move(out_fil_sub_1_temp_up, out_fil_sub_1)
                                    G_sub_1 = g_loop_sub_1_up

                        Q_sub_1 = q_loop_sub_1
                        P_sub_1 = P_new_sub_1
                         
                        time_step = time_step_sub_1
                        energy_dict_sub_1_new = self.mdif.state_energy(out_fil_sub_1,False)

                        reset_windows(
                            (time_step_list, q_list, g_list, p_list, steps_energy_dict),
                            (time_step, Q_sub_1, G_sub_1, P_sub_1, energy_dict_sub_1_new),
                        )

                        Q = self.cal_X(Q_sub_1,P_sub_1,G_sub_1,time_step)
                        self.mdif.replace_coordinate(inp_fil_sub_1,inp_fil_current,Q)
                        self.mdif.run(inp_fil_current)
                        check_result = self.mdif.check_out(out_fil_current)

                        if check_result != None:
                            print("DENSITY OR ENERGY CONVERGED")
                            print("Ab initial recalculation is convergence!")
                            G = self.mdif.get_gradient(out_fil_current)
                            P_ensemble = self.te.ensemble(momentum=P_sub_1, loop=loop, nsw=nsw)
                            P = self.cal_P(G_sub_1,G,P_ensemble,time_step)
                            Temperature = self.te.get_temperature(P)
                            time_step_list.append(time_step)
                        elif check_result == None:
                            print("Ab initial recalculation is misconvergence!")
                            print('\n{0:*^63}'.format(" Start Back Recalculation! "))
                            Q,back_time_step = self.back_cal(inp_fil_current,out_fil_current,inp_fil_sub_1,Q_sub_1,P_sub_1,G_sub_1)
                            G = self.mdif.get_gradient(out_fil_current)
                            P_ensemble = self.te.ensemble(momentum=P_sub_1, loop=loop, nsw=nsw)
                            P = self.cal_P(G_sub_1,G,P_ensemble,back_time_step)
                            Temperature = self.te.get_temperature(P)
                            time_step = back_time_step
                            time_step_list.append(1-time_step)
                            
                        current_time = sub_1_time + time_step
                        append_windows(
                            (q_list, g_list, p_list, steps_energy_dict),
                            (Q, G, P, energy_dict),
                        )

                        self.mdif.print_matrix(Q*self.unit.au_to_ang,'Recalculate Coordinate',str(loop),current_time)
                        self.mdif.print_matrix(G,'Recalculate Gradient',str(loop),current_time)
                        self.mdif.print_matrix(P,'Recalculate Momentun',str(loop),current_time)                          

                        kin_energy = self.cal_E_Kin(P)
                        current_state = self.mdif.get_state(out_fil_current)
                        energy_dict = self.mdif.state_energy(out_fil_current,False)
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

                        steps_energy_dict.append(energy_dict)

                        if h_s == 'U-D':
                            if v_factor_sub_1 == -1:
                                print("Forbidden transitions, no hopping!")
                            else:
                                print("step:{0} time:{1:.2f} hopping event:{2} -> {3}".format(str(loop),current_time,state_u,state_d))
                                self.write_matrix("ConIs.xyz",Q*self.unit.au_to_ang,loop,current_time)
                                with open("hopping_flag.log","a+") as f:
                                    f.write("step:{0} time:{1:.1f} hopping event:{2} -> {3}\n".format(str(loop),current_time,state_u,state_d))
                        elif h_s == 'D-U':
                            if v_factor_sub_1 == -1:
                                print("Forbidden transitions, no hopping!")
                            else:
                                print("step:{0} time:{1:.2f} hopping event:{2} -> {3}".format(str(loop),current_time,state_d,state_u))
                                self.write_matrix("ConIs.xyz",Q*self.unit.au_to_ang,loop,current_time)
                                with open("hopping_flag.log","a+") as f:
                                    f.write("step:{0} time:{1:.1f} hopping event:{2} -> {3}\n".format(str(loop),current_time,state_d,state_u))
                        elif h_s == "Doub_Hop":
                            if judge_label == "current_to_lower":
                                if v_factor_sub_1 == -1:
                                    print("Forbidden transitions, no hopping!")
                                else:
                                    print("step:{0} time:{1:.2f} hopping event:{2} -> {3}".format(str(loop),current_time,state_c,state_d))
                                    self.write_matrix("ConIs.xyz",Q*self.unit.au_to_ang,loop,current_time)
                                    with open("hopping_flag.log","a+") as f:
                                        f.write("step:{0} time:{1:.1f} hopping event:{2} -> {3}\n".format(str(loop),current_time,state_c,state_d))
                            elif judge_label == "current_to_upper":
                                if v_factor_sub_1 == -1:
                                    print("Forbidden transitions, no hopping!")
                                else:
                                    print("step:{0} time:{1:.2f} hopping event:{2} -> {3}".format(str(loop),current_time,state_c,state_u))
                                    self.write_matrix("ConIs.xyz",Q*self.unit.au_to_ang,loop,current_time)
                                    with open("hopping_flag.log","a+") as f:
                                        f.write("step:{0} time:{1:.1f} hopping event:{2} -> {3}\n".format(str(loop),current_time,state_c,state_u))
                    else:
                        print("Hopping failure!")
                        if h_s in ('U-D', 'D-U'):
                            self.mdif.remove_file(prefix_sub_1_new)
                        elif h_s == "Doub_Hop":
                            if judge_label == "current_to_lower":
                                self.mdif.remove_file(prefix_sub_1_down)
                            elif judge_label == "current_to_upper":
                                self.mdif.remove_file(prefix_sub_1_up)

            if loop>=3:
                prefix = self.init_input.split(".")[0]+"_"+str(loop-2)
                self.mdif.remove_file(prefix)

            self.write_matrix("coordinate.xyz",Q*self.unit.au_to_ang,loop,current_time)
            self.write_matrix("grdient.log",G,loop,current_time)
            self.write_matrix("momentum.log",P,loop,current_time)
            self.write_energy("energy.log",[E_tot,energy_current_state,kin_energy] + energy_list,loop,current_time)
            self.write_temperature("temperature.log",Temperature,loop,current_time)
            self.write_state("state.log",int(current_state),loop,current_time)

            loop = loop + 1
            time_step = time_step_list[-1]
        
        end_time = time.time()
        print('\n{0:*^63}'.format(" Time consuming "))
        print("Time consuming:\t{0}(seconds)".format(end_time - start_time))


