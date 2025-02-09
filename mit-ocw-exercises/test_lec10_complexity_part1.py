import lec10_complexity_part1 as lec10
import pytest

# create a fixture for the empty list
@pytest.fixture
def empty_list():
    return []

# create a fixture for the list of integers
@pytest.fixture
def sample_data():
    data = [1, 2, 3, 4, 5]
    return data

def test_linear_search():
    assert lec10.linear_search([1, 2, 3, 4, 5], 3) == True 
    assert lec10.linear_search([1, 2, 3, 4, 5], 6) == False
    assert lec10.linear_search([], 3) == False

def test_linear_search_int_list(sample_data):
    assert lec10.linear_search(sample_data, 3) == True 
    assert lec10.linear_search(sample_data, 6) == False

# create a list of integers, with parameter n for number of elements
def int_list(n):
    return list(range(n))

def test_linear_search_int_list():
    assert lec10.linear_search(int_list(10), 4) == True
    assert lec10.linear_search(int_list(10000), 4) == True
    #assert lec10.linear_search(int_list(100000000), 4) == True

# create a fixture factory that takes a parameter n for number of elements
@pytest.fixture
def int_list_factory():
    def _int_list(n):
        return list(range(n))
    return _int_list

def test_linear_search_int_list_factory(int_list_factory):
    assert lec10.linear_search(int_list_factory(10), 4) == True
    assert lec10.linear_search(int_list_factory(10000), 4) == True
    assert lec10.linear_search(int_list_factory(100000000), 4) == True

# create a fixture for num_elements with parameters 10, 10000, 1000000
@pytest.fixture(params=[10, 10000, 10000000])
def num_elements(request):
    return request.param

def test_linear_search_int_list_factory_enum(int_list_factory, num_elements):
    assert lec10.linear_search(int_list_factory(num_elements), 4) == True

# test intersection
def test_intersect():
    assert lec10.intersect([1, 2, 3, 2, 3], [2, 3, 4]) == [2, 3]
    assert lec10.intersect([1, 2, 3], [4, 5, 6]) == []
    assert lec10.intersect([], [4, 5, 6]) == []

# check elements in a list via python built-in function
def test_intersect_in():
    assert 2 in [1, 2, 3]
    assert 4 not in [1, 2, 3]



def test_intersect_additional():
    assert lec10.intersect([1, 2, 3, 4, 5], [4, 5, 6]) == [4, 5]




