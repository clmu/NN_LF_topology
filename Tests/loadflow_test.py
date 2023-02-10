import LF_3bus.ElkLoadFlow as elkLF
import LF_3bus.build_sys_from_sheet as build

BusList, LineList = build.BuildSystem('/home/clemens/PycharmProjects/NN_LF_Topology/LF_3bus/4 bus 1 gen.xls') #import from excel
bus4 = elkLF.decoupled_LF(BusList, LineList) #create LF object
