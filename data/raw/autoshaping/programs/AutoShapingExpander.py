"""
Script that expands AutoShaping.MPC into six different versions
"""
def expand_MPC(output_file,interval_list,probability_left):
	"""
	Examples of valid arguments:
	interval_list='LIST I = 20,25,30,35,40,45,50,55,60,65,70'
	probability_left='P = 5000'
	output_file='exit_file.MPC'
	"""
	with open('AutoShaping.MPC','r') as base:
		data=base.readlines()
	data[57]=interval_list+'\r\n'
	data[100]='\tset '+probability_left+';\r\n'
	with open(output_file,'w') as modified:
		modified.writelines(data)

#expand_MPC('atsh_LnR_Cnstnt.MPC','LIST I = 45','P = 5000')
#expand_MPC('atsh_LnR_Vrbl.MPC','LIST I = 20,25,30,35,40,45,50,55,60,65,70','P = 5000')
#expand_MPC('atsh_L_Cnstnt.MPC','LIST I = 45','P = 10000')
#expand_MPC('atsh_L_Vrbl.MPC','LIST I = 20,25,30,35,40,45,50,55,60,65,70','P = 10000')
#expand_MPC('atsh_R_Cnstnt.MPC','LIST I = 45','P = 0')
#expand_MPC('atsh_R_Vrbl.MPC','LIST I = 20,25,30,35,40,45,50,55,60,65,70','P = 0')

def expand_MPC_VR(output_file,interval_list,probability_left):
	"""
	Examples of valid arguments:
	interval_list='LIST I = 20,25,30,35,40,45,50,55,60,65,70'
	probability_left='P = 5000'
	output_file='exit_file.MPC'
	"""
	with open('AutoShaping_VR.MPC','r') as base:
		data=base.readlines()
	data[57]=interval_list+'\r\n'
	data[100]='\tset '+probability_left+';\r\n'
	with open(output_file,'w') as modified:
		modified.writelines(data)

#expand_MPC('atsh_VR_LnR_Vrbl.MPC','LIST I = 20,25,30,35,40,45,50,55,60,65,70','P = 5000')
expand_MPC_VR('atsh_VR_LnR_Vrbl.MPC','LIST I = 20,25,30,35,40,45,50','P = 5000')
