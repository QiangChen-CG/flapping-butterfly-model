import pickle
from buildbutterfly import *
import buildbutterfly

# import pre-built butterfly
bf_file = open('treeNymph.pkl', 'rb')
bf = pickle.load(bf_file)

new_head = buildbutterfly.BodySphere(0.02, 0.006)
new_head_root = np.array([0, 1, 0])

bf.add_body_part('new_head', new_head, new_head_root)

print(bf.body.new_head)
print(bf.root.new_head)