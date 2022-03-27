from lexnet.utils import relu, sigmoid, drelu, dsigmoid 

def test_relu():
    assert relu(-1)[0, 0] == 0
    assert relu(1)[0, 0] == 1
    assert relu([-1, 1])[0, 0] == 0
    assert relu([-1, 1])[0, 1] == 1
    assert relu([[1, 2], [-1, -2]])[1,1] == 0  
    assert relu([[1, 2], [-1, -2]])[1,0] == 0  
    assert relu([[1, 2], [-1, -2]])[0,0] == 1    
    assert relu([[1, 2], [-1, -2]])[0,1] == 2

def test_sigmoid():
    assert sigmoid(0)[0, 0] == 0.5
    assert sigmoid([0, 0, 0])[0, 0] == 0.5