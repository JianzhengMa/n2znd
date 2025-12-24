"""
Gradient recalculation helpers for Zhu–Nakamura workflow.
"""


class GradientMixin:
    def cal_single_adjacent_gradient(self,inp_fil_sub_2,inp_fil_sub_1,inp_fil_current,h_s,state_u,state_d):
        prefix_sub_2_new = inp_fil_sub_2.split(".")[0]+"_new"
        prefix_sub_1_new = inp_fil_sub_1.split(".")[0]+"_new"
        prefix_current_new = inp_fil_current.split(".")[0]+"_new"

        inp_fil_sub_2_temp_new,out_fil_sub_2_temp_new = self.mdif.gen_filename(prefix_sub_2_new)
        inp_fil_sub_1_temp_new,out_fil_sub_1_temp_new = self.mdif.gen_filename(prefix_sub_1_new)
        inp_fil_current_temp_new,out_fil_current_temp_new = self.mdif.gen_filename(prefix_current_new)

        if h_s == 'U-D':
            self.mdif.replace_state(inp_fil_sub_2,inp_fil_sub_2_temp_new,state_d)
            self.mdif.replace_state(inp_fil_sub_1,inp_fil_sub_1_temp_new,state_d)
            self.mdif.replace_state(inp_fil_current,inp_fil_current_temp_new,state_d)

            print("\nCalculate the gradient of lower state {0} for current steps-2".format(state_d))
            self.mdif.run(inp_fil_sub_2_temp_new)
            check_result_sub_2 = self.mdif.check_out(out_fil_sub_2_temp_new)
            if check_result_sub_2 != None:
                print("DENSITY OR ENERGY CONVERGED")
                print("Ab initial recalculation is convergence!")
            elif check_result_sub_2 == None:
                print("Ab initial recalculation is misconvergence!")

            print("\nCalculate the gradient of lower state {0} for current steps-1".format(state_d))
            self.mdif.run(inp_fil_sub_1_temp_new)
            check_result_sub_1 = self.mdif.check_out(out_fil_sub_1_temp_new)
            if check_result_sub_1 != None:
                print("DENSITY OR ENERGY CONVERGED")
                print("Ab initial recalculation is convergence!")
            elif check_result_sub_1 == None:
                print("Ab initial recalculation is misconvergence!")
            
            print("\nCalculate the gradient of lower state {0} for current steps".format(state_d))
            self.mdif.run(inp_fil_current_temp_new)
            check_result_current = self.mdif.check_out(out_fil_current_temp_new)
            if check_result_current != None:
                print("DENSITY OR ENERGY CONVERGED")
                print("Ab initial recalculation is convergence!")
            elif check_result_current == None:
                print("Ab initial recalculation is misconvergence!")

        elif h_s == 'D-U':
            self.mdif.replace_state(inp_fil_sub_2,inp_fil_sub_2_temp_new,state_u)
            self.mdif.replace_state(inp_fil_sub_1,inp_fil_sub_1_temp_new,state_u)
            self.mdif.replace_state(inp_fil_current,inp_fil_current_temp_new,state_u)

            print("\nCalculate the gradient of upper state {0} for current steps-2".format(state_u))         
            self.mdif.run(inp_fil_sub_2_temp_new)
            check_result_sub_2 = self.mdif.check_out(out_fil_sub_2_temp_new)
            if check_result_sub_2 != None:
                print("DENSITY OR ENERGY CONVERGED")
                print("Ab initial recalculation is convergence!")
            elif check_result_sub_2 == None:
                print("Ab initial recalculation is misconvergence!")

            print("\nCalculate the gradient of upper state {0} for current steps-1".format(state_u))
            self.mdif.run(inp_fil_sub_1_temp_new)
            check_result_sub_1 = self.mdif.check_out(out_fil_sub_1_temp_new)
            if check_result_sub_1 != None:
                print("DENSITY OR ENERGY CONVERGED")
                print("Ab initial recalculation is convergence!")
            elif check_result_sub_1 == None:
                print("Ab initial recalculation is misconvergence!")

            print("\nCalculate the gradient of upper state {0} for current steps".format(state_u))
            self.mdif.run(inp_fil_current_temp_new)
            check_result_current = self.mdif.check_out(out_fil_current_temp_new)
            if check_result_current != None:
                print("DENSITY OR ENERGY CONVERGED")
                print("Ab initial recalculation is convergence!")
            elif check_result_current == None:
                print("Ab initial recalculation is misconvergence!")

        g_loop_sub_2_new = self.mdif.get_gradient(out_fil_sub_2_temp_new)
        g_loop_sub_1_new = self.mdif.get_gradient(out_fil_sub_1_temp_new)
        g_loop_current_new = self.mdif.get_gradient(out_fil_current_temp_new)
        self.mdif.remove_file(prefix_sub_2_new)
        self.mdif.remove_file(prefix_current_new)
        return g_loop_sub_2_new, g_loop_sub_1_new, g_loop_current_new, \
                inp_fil_sub_1_temp_new, out_fil_sub_1_temp_new, prefix_sub_1_new

    def cal_double_adjacent_gradient(self,inp_fil_sub_2,inp_fil_sub_1,inp_fil_current,state_u,state_d):
        prefix_sub_2_up = inp_fil_sub_2.split(".")[0]+"_up"
        prefix_sub_1_up = inp_fil_sub_1.split(".")[0]+"_up"
        prefix_current_up = inp_fil_current.split(".")[0]+"_up"

        prefix_sub_2_down = inp_fil_sub_2.split(".")[0]+"_down"
        prefix_sub_1_down = inp_fil_sub_1.split(".")[0]+"_down"
        prefix_current_down = inp_fil_current.split(".")[0]+"_down"

        inp_fil_sub_2_temp_up,out_fil_sub_2_temp_up = self.mdif.gen_filename(prefix_sub_2_up)
        inp_fil_sub_1_temp_up,out_fil_sub_1_temp_up = self.mdif.gen_filename(prefix_sub_1_up)
        inp_fil_current_temp_up,out_fil_current_temp_up = self.mdif.gen_filename(prefix_current_up)

        inp_fil_sub_2_temp_down,out_fil_sub_2_temp_down = self.mdif.gen_filename(prefix_sub_2_down)
        inp_fil_sub_1_temp_down,out_fil_sub_1_temp_down = self.mdif.gen_filename(prefix_sub_1_down)
        inp_fil_current_temp_down,out_fil_current_temp_down = self.mdif.gen_filename(prefix_current_down)

        self.mdif.replace_state(inp_fil_sub_2,inp_fil_sub_2_temp_up,state_u)
        self.mdif.replace_state(inp_fil_sub_1,inp_fil_sub_1_temp_up,state_u)
        self.mdif.replace_state(inp_fil_current,inp_fil_current_temp_up,state_u)

        self.mdif.replace_state(inp_fil_sub_2,inp_fil_sub_2_temp_down,state_d)
        self.mdif.replace_state(inp_fil_sub_1,inp_fil_sub_1_temp_down,state_d)
        self.mdif.replace_state(inp_fil_current,inp_fil_current_temp_down,state_d)

        print("\nCalculate the gradient of lower state {0} for current steps-2".format(state_u))
        self.mdif.run(inp_fil_sub_2_temp_up)
        check_result_sub_2_up = self.mdif.check_out(out_fil_sub_2_temp_up)
        if check_result_sub_2_up != None:
            print("DENSITY OR ENERGY CONVERGED")
            print("Ab initial recalculation is convergence!")
        elif check_result_sub_2_up == None:
            print("Ab initial recalculation is misconvergence!")

        print("\nCalculate the gradient of lower state {0} for current steps-1".format(state_u))
        self.mdif.run(inp_fil_sub_1_temp_up)
        check_result_sub_1_up = self.mdif.check_out(out_fil_sub_1_temp_up)
        if check_result_sub_1_up != None:
            print("DENSITY OR ENERGY CONVERGED")
            print("Ab initial recalculation is convergence!")
        elif check_result_sub_1_up == None:
            print("Ab initial recalculation is misconvergence!")
        
        print("\nCalculate the gradient of lower state {0} for current steps".format(state_u))
        self.mdif.run(inp_fil_current_temp_up)
        check_result_current_up = self.mdif.check_out(out_fil_current_temp_up)
        if check_result_current_up != None:
            print("DENSITY OR ENERGY CONVERGED")
            print("Ab initial recalculation is convergence!")
        elif check_result_current_up == None:
            print("Ab initial recalculation is misconvergence!")

        print("\nCalculate the gradient of lower state {0} for current steps-2".format(state_d))
        self.mdif.run(inp_fil_sub_2_temp_down)
        check_result_sub_2_down = self.mdif.check_out(out_fil_sub_2_temp_down)
        if check_result_sub_2_down != None:
            print("DENSITY OR ENERGY CONVERGED")
            print("Ab initial recalculation is convergence!")
        elif check_result_sub_2_down == None:
            print("Ab initial recalculation is misconvergence!")

        print("\nCalculate the gradient of lower state {0} for current steps-1".format(state_d))
        self.mdif.run(inp_fil_sub_1_temp_down)
        check_result_sub_1_down = self.mdif.check_out(out_fil_sub_1_temp_down)
        if check_result_sub_1_down != None:
            print("DENSITY OR ENERGY CONVERGED")
            print("Ab initial recalculation is convergence!")
        elif check_result_sub_1_down == None:
            print("Ab initial recalculation is misconvergence!")
        
        print("\nCalculate the gradient of lower state {0} for current steps".format(state_d))
        self.mdif.run(inp_fil_current_temp_down)
        check_result_current_down = self.mdif.check_out(out_fil_current_temp_down)
        if check_result_current_down != None:
            print("DENSITY OR ENERGY CONVERGED")
            print("Ab initial recalculation is convergence!")
        elif check_result_current_down == None:
            print("Ab initial recalculation is misconvergence!")

        g_loop_sub_2_up = self.mdif.get_gradient(out_fil_sub_2_temp_up)
        g_loop_sub_1_up = self.mdif.get_gradient(out_fil_sub_1_temp_up)
        g_loop_current_up = self.mdif.get_gradient(out_fil_current_temp_up)

        g_loop_sub_2_down = self.mdif.get_gradient(out_fil_sub_2_temp_down)
        g_loop_sub_1_down = self.mdif.get_gradient(out_fil_sub_1_temp_down)
        g_loop_current_down = self.mdif.get_gradient(out_fil_current_temp_down)

        self.mdif.remove_file(prefix_sub_2_up)
        self.mdif.remove_file(prefix_current_up)
        self.mdif.remove_file(prefix_sub_2_down)
        self.mdif.remove_file(prefix_current_down)
        return g_loop_sub_2_up,g_loop_sub_1_up,g_loop_current_up,\
                inp_fil_sub_1_temp_up,out_fil_sub_1_temp_up,prefix_sub_1_up,\
                    g_loop_sub_2_down,g_loop_sub_1_down,g_loop_current_down,\
                        inp_fil_sub_1_temp_down,out_fil_sub_1_temp_down,prefix_sub_1_down
