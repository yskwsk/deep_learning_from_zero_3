# import unittest

# import numpy as np

# from dezero import Variable


# class TestAdd(unittest.TestCase):
#     def test_add1(self):
#         x0 = Variable(np.array(1.0))
#         x1 = 2.0
#         y = x0 + x1
#         expected = np.array(3.0)
#         self.assertEqual(y.data, expected)

#     def test_add2(self):
#         x0 = 2.0
#         x1 = Variable(np.array(4.0))
#         y = x0 + x1
#         expected = np.array(6.0)
#         self.assertEqual(y.data, expected)

#     def test_add3(self):
#         x0 = Variable(np.array(5.0))
#         x1 = np.array(4.0)
#         y = x0 + x1
#         expected = np.array(9.0)
#         self.assertEqual(y.data, expected)

#     def test_add4(self):
#         x0 = np.array(2.0)
#         x1 = Variable(np.array(3.0))
#         y = x0 + x1
#         expected = np.array(5.0)
#         self.assertEqual(y.data, expected)

#     def test_add5(self):
#         x0 = Variable(np.array(6.5))
#         x1 = Variable(np.array(3.5))
#         y = x0 + x1
#         expected = np.array(10.0)
#         self.assertEqual(y.data, expected)


# class TestMul(unittest.TestCase):
#     def test_mul1(self):
#         x0 = Variable(np.array(2.0))
#         x1 = 3.0
#         y = x0 * x1
#         expected = np.array(6.0)
#         self.assertEqual(y.data, expected)

#     def test_mul2(self):
#         x0 = 4.0
#         x1 = Variable(np.array(3.0))
#         y = x0 * x1
#         expected = np.array(12.0)
#         self.assertEqual(y.data, expected)

#     def test_mul3(self):
#         x0 = Variable(np.array(6.0))
#         x1 = np.array(4.0)
#         y = x0 * x1
#         expected = np.array(24.0)
#         self.assertEqual(y.data, expected)

#     def test_mul4(self):
#         x0 = np.array(5.0)
#         x1 = Variable(np.array(8.0))
#         y = x0 * x1
#         expected = np.array(40.0)
#         self.assertEqual(y.data, expected)

#     def test_mul5(self):
#         x0 = Variable(np.array(9.0))
#         x1 = Variable(np.array(-5.0))
#         y = x0 * x1
#         expected = np.array(-45.0)
#         self.assertEqual(y.data, expected)


# class TestNeg(unittest.TestCase):
#     def test_neg1(self):
#         x = Variable(np.array(2.0))
#         y = -x
#         expected = np.array(-2.0)
#         self.assertEqual(y.data, expected)

#     def test_neg2(self):
#         x = Variable(np.array(-3))
#         y = -x
#         expected = np.array(3)
#         self.assertEqual(y.data, expected)


# class TestSub(unittest.TestCase):
#     def test_sub1(self):
#         x0 = Variable(np.array(3.0))
#         x1 = 1.0
#         y = x0 - x1
#         expected = np.array(2.0)
#         self.assertEqual(y.data, expected)

#     def test_sub2(self):
#         x0 = 1.0
#         x1 = Variable(np.array(5.0))
#         y = x0 - x1
#         expected = np.array(-4.0)
#         self.assertEqual(y.data, expected)

#     def test_sub3(self):
#         x0 = Variable(np.array(9.0))
#         x1 = np.array(4.0)
#         y = x0 - x1
#         expected = np.array(5.0)
#         self.assertEqual(y.data, expected)

#     def test_sub4(self):
#         x0 = np.array(-4.0)
#         x1 = Variable(np.array(6.0))
#         y = x0 - x1
#         expected = np.array(-10.0)
#         self.assertEqual(y.data, expected)

#     def test_sub5(self):
#         x0 = Variable(np.array(10.0))
#         x1 = Variable(np.array(5.0))
#         y = x0 - x1
#         expected = np.array(5.0)
#         self.assertEqual(y.data, expected)


# class TestDiv(unittest.TestCase):
#     def test_div1(self):
#         x0 = Variable(np.array(2.0))
#         x1 = 4.0
#         y = x0 / x1
#         expected = np.array(0.5)
#         self.assertEqual(y.data, expected)

#     def test_div2(self):
#         x0 = 4.0
#         x1 = Variable(np.array(2.0))
#         y = x0 / x1
#         expected = np.array(2.0)
#         self.assertEqual(y.data, expected)

#     def test_div3(self):
#         x0 = Variable(np.array(2.0))
#         x1 = np.array(4.0)
#         y = x0 / x1
#         expected = np.array(0.5)
#         self.assertEqual(y.data, expected)

#     def test_div4(self):
#         x0 = np.array(4.0)
#         x1 = Variable(np.array(2.0))
#         y = x0 / x1
#         expected = np.array(2.0)
#         self.assertEqual(y.data, expected)

#     def test_div5(self):
#         x0 = Variable(np.array(25.0))
#         x1 = Variable(np.array(2.0))
#         y = x0 / x1
#         expected = np.array(12.5)
#         self.assertEqual(y.data, expected)


# class TestPow(unittest.TestCase):
#     def test_pow1(self):
#         x0 = Variable(np.array(2.0))
#         x1 = 3.0
#         y = x0 ** x1
#         expected = np.array(8.0)
#         self.assertEqual(y.data, expected)

#     def test_pow2(self):
#         x0 = 3.0
#         x1 = Variable(np.array(4.0))
#         y = x0 ** x1
#         expected = np.array(81.0)
#         self.assertEqual(y.data, expected)

#     def test_pow3(self):
#         x0 = Variable(np.array(4.0))
#         x1 = np.array(3.0)
#         y = x0 ** x1
#         expected = np.array(64.0)
#         self.assertEqual(y.data, expected)

#     def test_pow4(self):
#         x0 = np.array(8.0)
#         x1 = Variable(np.array(2.0))
#         y = x0 ** x1
#         expected = np.array(64.0)
#         self.assertEqual(y.data, expected)

#     def test_pow5(self):
#         x0 = Variable(np.array(5.0))
#         x1 = Variable(np.array(2.0))
#         y = x0 ** x1
#         expected = np.array(25.0)
#         self.assertEqual(y.data, expected)
