#bit manipulation function to apply to reservoir weights matrix after overflow issue is resolved
# for testing bfloat16, regular float 16 we can just call simple numpy function
import numpy as np
q = np.float32(-8.56778)
def Binarator(x):
    #Function Takes float, returns binary string
    #bfloat 16 is simply just regular float 32 with 16 fewer bits in mantissa
    #thus all we need to do is change the last 16 bits to 0
    # first let's get the binary representation
        # Declaring an empty string
    # to store binary bits.
    print(type(x))

    
 
    # Setting Sign bit
    # default to zero.
    sign_bit = 0
    # Sign bit will set to
    # 1 for negative no.
    if(x < 0):
        sign_bit = 1
    x = abs(x)
    #change mantissa to bin

    #integer part of mantissa
    int_str = bin(int(x))[2 : ]
    #fraction part of mantissa
    fraction = (x - int(x))
    fraction_str  = str()
    
    while (fraction):
        fraction *= 2
        if (fraction >= 1):
            int_part = 1
            fraction -= 1
        else:
            int_part = 0
     
        
        #recombine mantissa
        fraction_str  += str(int_part)
    #combine binary string
    ind = int_str.index('1')
    exp_str = bin((len(int_str) - ind - 1) + 127)[2 : ]
    mant_str = int_str[ind + 1 : ] + fraction_str
    mant_str = mant_str + ('0' * (23 - len(mant_str)))
 
    binary = str(sign_bit) + exp_str + mant_str
    
    print(binary)
    print(str(sign_bit))
    print(exp_str) 
    print(mant_str)
    return binary

def f32tob16(x):
    #this function takes as input a float16 at outputs an analogue for Bfloat 16
    #we cant actually change to Bfloat 16, so we emulate it by changing the last16 bits of float32 to zero
    pq = list(Binarator(x))
    for i in range(16, 32):
        pq[i] = '0'
    q = ""
    for i in pq:
        q+=i
    p = ((-1)**int(q[0])) * (2**(int(q[1:9],2)-127)) * (1 + (int(q[9:],2)*(2**-23)))
    return p
#quick test
print(f32tob16(q))