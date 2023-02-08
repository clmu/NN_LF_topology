import pandapower as pp
#create empty net
net = pp.networks.case4gs()

pp.runpp(net)