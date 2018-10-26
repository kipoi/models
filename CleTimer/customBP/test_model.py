import unittest
import random
import model
import numpy as np


class ModelTest(unittest.TestCase):

    def setUp(self):
        self.soi_g80 = []
        for i in range(10):  # generate test sequences
            length = random.randint(101, 1000)
            self.soi_g80.append(''.join(random.choices(['A', 'C', 'G', 'T', 'a', 'c', 'g', 't'], k=length)))
        self.soi_g80 = np.array(self.soi_g80, dtype=np.string_)
        self.features_metadata = model.load_features_metadata()

    def testRandom_baseFeaturesConstruction(self):
        """
        Reverse test the features array getting created properly from sequences.
        """
        mod = model.CleavageTimeModel()
        features_array = mod._construct_features_array(self.soi_g80)
        print("Shape of features array: ", features_array.shape)

        self.bp_indexes = mod.bp_indexes
        self.acc_i = mod.acc_i
        self.don_i = mod_don_i

        soi = self.soi_g80
        for j in range(len(self.soi_g80)):

            for i in range(2, len(self.features_metadata)):

                control_predicate = int(features_array[j, i]) == 1

                (region, pos, nucl) = self.features_metadata[i]

                if region == 'seqB':
                    i_oi = int(self.bp_indexes[j]) + int(pos)
                    target_predicate = soi[j][i_oi].upper() == nucl
                else:
                    if pos > 0:  # decrement, since acc_i is pos = 1
                        pos -= 1

                    if region == 'seqA':
                        target_predicate = soi[j][(acc_i + int(pos))].upper() == nucl
                        pos = acc_i + int(pos)

                    elif region == 'seqD':
                        target_predicate = soi[j][(don_i + int(pos))].upper() == nucl
                        pos = don_i + int(pos)

                message = ''.join(["Comparing feature ", str(i), " in sequence ", soi[j], ": got ", str(features_array[j, i]), " in region ", str(region), " at index ", str(pos),
                                   ",  and the nucleotide ", nucl])
                self.assertEqual(control_predicate, target_predicate, msg=message)

            print(''.join(["Got features ", str(features_array[j])]))

            # check branchpoint features
            bp_features_indexes = [index for (index, x) in enumerate(self.features_metadata) if x[0] == 'seqB']
            debug = ''.join(["Branchpoint features ", str(features_array[j][bp_features_indexes]), " wtih branchpoint on index ", str(self.bp_indexes[j]), " for sequence ", soi[j]])
            print(debug)

#     def test_rejectsShortIntrons(self):
#         mod = model.CleavageTimeModel()
#         self.assertRaises(ValueError, mod.predict_on_batch, "AgctGAcAtt")
