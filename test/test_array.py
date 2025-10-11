import mininumpy as mnp

def test_array_data():
    assert mnp.array([[1, 2, 3], [4, 5, 6], [7, 0, 9]]).data == [[1, 2, 3], [4, 5, 6], [7, 0, 9]] 