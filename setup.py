from tables import *

class element(IsDescription):
    Community = Float64Col(shape=(462,))
    Lambda = Float64Col(shape=(106491,))

h5file = open_file('E:/train.h5', mode='w')
group = h5file.create_group("/", 'Group1', 'My Group')
table = h5file.create_table(group, 'Train', element, "Train Data")
table.flush()
h5file.close()
