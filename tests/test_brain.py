import unittest
from mrilab.brain import Brain


class TestBrain(unittest.TestCase):

	b = Brain('tests/data/img_1.nii')
	
	def test_init(self):
		self.assertEqual(self.b.shape, (176, 208, 176, 1))
