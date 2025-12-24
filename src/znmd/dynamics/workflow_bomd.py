import os
import time
import random
import numpy as np


class BOMDWorkflowMixin:
    # Born-Oppenheimer approximation molecular dynamics
    def bomd_on_the_fly(self):
        start_time = time.time()
        print("Start Born-Oppenheimer Molecular Dynamics Simulation")
        print("Localtime:",time.asctime(time.localtime(time.time())))
        print("Current work dictory is:%s\n"%(os.getcwd()))
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

        self.mdif.clear_file()
        loop = 0
        time_step = self.time_step
        current_time  = 0.0
        nsw = int(self.total_time/self.time_step)

        while loop < nsw:
            
            if loop == 0:
                Q_old = self.mdif.get_coordinate(self.init_input)
                input_file = self.init_input
                output_file = self.init_input.split('.')[0] + ".out"
                self.mdif.run(input_file)
                check_result = self.mdif.check_out(output_file)
                print('{0:-^63}'.format("-"))
                print("-----------------------Output: step = {0}--------------------------".format(loop))
                print('{0:-^63}\n'.format("-"))

                if check_result != None:
                    print("DENSITY OR ENERGY CONVERGED")
                    print("Ab initial calculation is convergence!")
                    G_old = self.mdif.get_gradient(output_file)
                    if os.path.exists(self.init_momentum):
                        P_old = self.mdif.ini_momentum(self.init_momentum)
                    else:
                        P_old = self.te.maxwell_boltzmann_momentum(self.initial_temperature)
                    Temperature = self.te.get_temperature(P_old)
                elif check_result == None:
                    print("Ab initial calculation is misconvergence!")
                    print("Changing the atomic coordinate might solve this problem!")
                    os._exit(0)

                current_state = self.mdif.get_state(output_file)
                energy_dict = self.mdif.state_energy(output_file,False)
                self._log_bomd_step(Q_old, G_old, P_old, energy_dict, current_state, Temperature, loop, current_time)
                loop = loop + 1

            elif loop >= 1:
                Q = self.cal_X(Q_old,P_old,G_old,time_step)
                prefix = self.init_input.split(".")[0]+"_"+str(loop)
                input_file,output_file = self.mdif.gen_filename(prefix)
                self.mdif.replace_coordinate(self.init_input,input_file,Q)
                
                self.mdif.run(input_file)
                check_result = self.mdif.check_out(output_file)
                print('{0:-^63}'.format("-"))
                print("-----------------------Output: step = {0}--------------------------".format(loop))
                print('{0:-^63}\n'.format("-"))

                if check_result != None:
                    print("DENSITY OR ENERGY CONVERGED")
                    print("Ab initial calculation is convergence!")
                    G = self.mdif.get_gradient(output_file)
                    P_ensemble = self.te.ensemble(momentum=P_old, loop=loop, nsw=nsw)
                    P = self.cal_P(G_old,G,P_ensemble,time_step)
                    Temperature = self.te.get_temperature(P)
                elif check_result == None:
                    print("Ab initial calculation is misconvergence!")
                    print('\n{0:*^63}'.format(" Start Back Recalculation! "))
                    Q,back_time_step = self.back_cal(input_file,output_file,self.init_input,Q_old,P_old,G_old)
                    G = self.mdif.get_gradient(output_file)
                    P_ensemble = self.te.ensemble(momentum=P_old, loop=loop, nsw=nsw)
                    P= self.cal_P(G_old,G,P_ensemble,back_time_step) 
                    Temperature = self.te.get_temperature(P)
                    time_step = back_time_step
            
                current_time = current_time + time_step
                current_state = self.mdif.get_state(output_file)
                energy_dict = self.mdif.state_energy(output_file,False)
                self._log_bomd_step(Q, G, P, energy_dict, current_state, Temperature, loop, current_time)

                Q_old = Q
                P_old = P
                G_old = G
                loop = loop + 1

                if check_result == None:
                    time_step = 1 - back_time_step

            if loop >= 2:
                prefix = self.init_input.split(".")[0]+"_"+str(loop-1)
                self.mdif.remove_file(prefix)
        
        end_time = time.time()
        print('\n{0:*^63}'.format(" Time consuming "))
        print("Time consuming:\t{0}(seconds)".format(end_time - start_time))


