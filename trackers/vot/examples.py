from Sequence import Sequence
from tools import Tracking

sequence = Sequence(path='/home/wnj/projects/Visual_Tracking_api-master', name='bicycle', region_format='rectangle')

Tracking(sequence,tracker_list=['KCFtracker'],visualize=True)

