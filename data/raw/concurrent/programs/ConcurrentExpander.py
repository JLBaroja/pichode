"""
Script that expands AutoShaping.MPC into six different versions

def expand_MPC(output_file,interval_list,probability_left):
	Examples of valid arguments:
	interval_list='LIST I = 20,25,30,35,40,45,50,55,60,65,70'
	probability_left='P = 5000'
	output_file='exit_file.MPC'
	with open('AutoShaping.MPC','r') as base:
		data=base.readlines()
	data[57]=interval_list+'\r\n'
	data[100]='\tset '+probability_left+';\r\n'
	with open(output_file,'w') as modified:
		modified.writelines(data)

expand_MPC('atsh_LnR_Cnstnt.MPC','LIST I = 45','P = 5000')
expand_MPC('atsh_LnR_Vrbl.MPC','LIST I = 20,25,30,35,40,45,50,55,60,65,70','P = 5000')
expand_MPC('atsh_L_Cnstnt.MPC','LIST I = 45','P = 10000')
expand_MPC('atsh_L_Vrbl.MPC','LIST I = 20,25,30,35,40,45,50,55,60,65,70','P = 10000')
expand_MPC('atsh_R_Cnstnt.MPC','LIST I = 45','P = 0')
expand_MPC('atsh_R_Vrbl.MPC','LIST I = 20,25,30,35,40,45,50,55,60,65,70','P = 0')
"""

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

def expand_concurrent(vi_left,
			vi_right,
			discrepancies=[0,0],
			lower_first=15,
			upper_first=20,
			lower_second=35,
			upper_second=40,
			session_length=60,
			file='default'):
	with open('Concurrent_VI_VI.MPC','r') as base:
		data=base.readlines()

	# VI Schedules in each key
#	vi_left=[10,5,25]
#	vi_right=[20,30,15]
	vi_L=np.array(vi_left)
	vi_R=np.array(vi_right)
	vi_L_list=10000/vi_L
	vi_R_list=10000/vi_R
	vi_L_list=vi_L_list.astype('int')
	vi_R_list=vi_R_list.astype('int')
	vi_L_l=vi_L_list.astype('str')
	vi_R_l=vi_R_list.astype('str')
	list_l='LIST L = '+vi_L_l[0]+','+vi_L_l[1]+','+vi_L_l[2]
	list_r='LIST R = '+vi_R_l[0]+','+vi_R_l[1]+','+vi_R_l[2]

	# Lower and Upper limits of possible change points
#	lower_first=10
#	upper_first=40
#	lower_second=50
#	upper_second=80
	lwf=lower_first*60
	upf=upper_first*60
	lws=lower_second*60
	ups=upper_second*60
	possible_1st_ch=np.linspace(lwf,upf,upf-lwf+1)
	possible_2nd_ch=np.linspace(lws,ups,ups-lws+1)
	string_1st=possible_1st_ch.astype('int').astype('str')
	string_2nd=possible_2nd_ch.astype('int').astype('str')
	list_m='LIST M = '+string_1st[0]
	list_n='LIST N = '+string_2nd[0]
	for ii in range(len(string_1st)-1):
		list_m=list_m+','+string_1st[ii+1]
	for ii in range(len(string_2nd)-1):
		list_n=list_n+','+string_2nd[ii+1]

	# Discrepancies
	discrepancies=np.array(discrepancies)
	dr=discrepancies.astype('str')
	list_d='LIST D = '+dr[0]+','+dr[1]

	# Session length
	variable_s=str(session_length*60*100)

	# Lines substitution
	data[data.index('\\\\ Line for LIST M\n')]=list_m+'\n'
	data[data.index('\\\\ Line for LIST N\n')]=list_n+'\n'
	data[data.index('\\\\ Line for LIST D\n')]=list_d+'\n'
	data[data.index('\\\\ Line for LIST L\n')]=list_l+'\n'
	data[data.index('\\\\ Line for LIST R\n')]=list_r+'\n'
	data[data.index('\\\\ Line for variable S\n')]='\tset S = '+variable_s+';\n'

	vi_L=vi_L.astype('int')
	vi_R=vi_R.astype('int')
	vi_L=vi_L.astype('str')
	vi_R=vi_R.astype('str')
	if file=='default':
		file_label='japede_L_'+vi_L[0]+'_'+vi_L[1]+'_'+vi_L[2]+'_'+'R_'+vi_R[0]+'_'+vi_R[1]+'_'+vi_R[2]+'.MPC'
	else:
		file_label=file

	with open(file_label,'w') as modified:
		modified.writelines(data)

#	return data

#expand_concurrent([30,30,30],[90,90,90])
#expand_concurrent([30,90,90],[90,30,30])
#expand_concurrent([90,90,90],[30,30,30])
#expand_concurrent([90,30,30],[30,90,90])
#expand_concurrent([90,45,45],[30,45,45])
#expand_concurrent([45,45,45],[45,45,45])


expand_concurrent([90,float('Inf'),float('Inf')],[30,float('Inf'),float('Inf')],lower_first=20,upper_first=30,file='japede_L_90_inf_inf_R_30_inf_inf.MPC')
expand_concurrent([135,float('Inf'),float('Inf')],[27,float('Inf'),float('Inf')],lower_first=20,upper_first=30,file='japede_L_135_inf_inf_R_27_inf_inf.MPC')
expand_concurrent([25,float('Inf'),float('Inf')],[225,float('Inf'),float('Inf')],lower_first=20,upper_first=30,file='japede_L_25_inf_inf_R_225_inf_inf.MPC')



rich_side=[24,25,27,30,35]
poor_side=[360,225,135,90,63]

rich_right=pd.DataFrame()
rich_left=pd.DataFrame()
rich_right['left']=poor_side
rich_right['right']=rich_side
rich_left['left']=rich_side
rich_left['right']=poor_side


dynamic_schedule=pd.DataFrame([],columns=['bird','session','med_program','date'])
np_seed=1
np.random.seed(np_seed)
sessions=np.arange(116,301,1)
row=0
for bird in ['p004','p054','p138','p510','p530','p736']:
	start_date="2017-10-22"
	start=datetime.strptime(start_date,'%Y-%m-%d')
	for ss in sessions:
		start=start+timedelta(days=1)
		if ss==116:
			start_left=45
			start_right=45
			#final_left=45
			#final_right=45
		else:
			start_left=final_left
			start_right=final_right
		change=np.random.binomial(1,0.5)
		if change==0:
			final_left=start_left
			final_right=start_right
		else:
			indx=np.random.choice(range(len(rich_left['left'])))
			if sum(rich_left['left']==start_left)==1: # If started at rich left
				final_left=rich_right.iloc[indx]['left']
				final_right=rich_right.iloc[indx]['right']
			else: # If started at rich right
				final_left=rich_left.iloc[indx]['left']
				final_right=rich_left.iloc[indx]['right']

		med_program_file='L_'+str(start_left)+2*('_'+str(final_left))+'_R_'+str(start_right)+2*('_'+str(final_right))
		date_label=str(start).split(' ')[0]
		dy_sch=pd.DataFrame([[bird,str(ss),med_program_file,date_label]],columns=['bird','session','med_program','date'])
		dynamic_schedule=dynamic_schedule.append(dy_sch,ignore_index=True)
		expand_concurrent([start_left,final_left,final_left],[start_right,final_right,final_right],lower_first=20,upper_first=40,lower_second=45,upper_second=50)
		row=row+1
		#print bird+'s'+str(ss)+str(change)+med_program_file+date_label
dynamic_schedule.to_csv('dynamic_schedule.csv')














