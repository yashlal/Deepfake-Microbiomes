from tables import *

class element(IsDescription):
    Community = Float64Col(shape=(849,))
    Lambda = Float64Col(shape=(849,849))

h5file = open_file('Data/train.h5', mode='w')
group = h5file.create_group("/", 'Group1', 'My Group')
table = h5file.create_table(group, 'Train', element, "Train Data")
table.flush()
h5file.close()
