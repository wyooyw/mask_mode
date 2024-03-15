import torch
from mask_mode import mask_mode

def test():
    tensor = torch.Tensor([[1,1,1,2,3,4,4,4,4],[5,5,5,5,5,3,3,3,3]]).to(torch.int8).cuda(0)
    mask = torch.Tensor([[1,1,1,1,1,1,1,0,0],[0,0,1,1,1,1,1,1,1]]).to(torch.int8).cuda(0)
    answer = torch.Tensor([1,3]).to(torch.int32).cuda(0)
    result = mask_mode(tensor,mask)
    assert (answer==result).all(), f"Test fail. Answer={str(answer)}, reuslt={str(result)}"
    print("Pass test.")

if __name__=="__main__":
    test()
    