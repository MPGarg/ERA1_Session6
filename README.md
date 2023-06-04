## Assignment Session-6 Part-1

Backpropagation for 3 layer Neural Network is explained in detail below.

Neural network is under consideration as shown below & is assigned with initial weights.
![image](https://user-images.githubusercontent.com/120099863/211860337-40ddc717-28f0-4ae7-8094-5c371e8c1652.png)

Input Nodes:
1. i1
2. i2

Hidden Layer Nodes:
1. h1 = w1\*i1 + w2\*i2
2. h2 = w3\*i1 + w4\*i2

Activation values for Hidden Layer (Sigmoid):
1. a_h1 = σ(h1) = 1/(1 + exp(-h1))
2. a_h2 = σ(h2) = 1/(1 + exp(-h2))

Output Layer Nodes:
1. o1 = w5\*a_h1 + w6\*a_h2
2. o2 = w7\*a_h1 + w8\*a_h2

Activation values for Output Layer (Sigmoid):
1. a_o1 = σ(o1) = 1/(1 + exp(-o1))
2. a_o2 = σ(o2) = 1/(1 + exp(-o2))

Error Calculations:
1. E1 = (1/2) \* (t1 - a_o1)^2
2. E2 = (1/2) \* (t2 - a_o2)^2
3. E_Total = E1 + E2

Backpropagation calcualtions start once we have output and error is calculated.
1. Calculation of gradient for w5
      - ∂E_total/∂w5 = ∂(E1+E2)/∂w5	
      - As E2 is constant wrt w5 thus only E1 remains
        - ∂E_total/∂w5 = ∂E1/∂w5
      - Simplifying equation further
        - ∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1\*∂a_o1/∂o1\*∂o1/∂w5
      - ∂E1/∂ao1 is calculated below
        - ∂E1/∂ao1 = ∂((1/2) \* (t1 - a_o1)^2)/∂a_o1 = (a_o1 - t1)			
      - ∂a_o1/∂o1 is calculated below (sigmoid differential)
        - ∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = a_o1 \* (1 - a_o1)			
      - ∂o1/∂w5 calculation	
        - ∂o1/∂w5 = a_h1	
      - Putting all pieces together we get:
        - ∂E_total/∂w5 = (a_01 - t1) \* a_o1 \* (1 - a_o1) \* a_h1	
        	
2. Similar calculations can be done for weights w6, w7 & w8. Here are the equations for them respectively:
      - ∂E_total/∂w6 = (a_01 - t1) \* a_o1 \* (1 - a_o1) \* a_h2
      - ∂E_total/∂w7 = (a_02 - t2) \* a_o2 \* (1 - a_o2) \* a_h1
      - ∂E_total/∂w8 = (a_02 - t2) \* a_o2 \* (1 - a_o2) \* a_h2
      
3. Gradient for a_h1 is summation of two values as this node is connected by two routes.
      - ∂E_total/∂a_h1 = ∂E1/∂a_h1 + ∂E2/∂a_h1
      - ∂E1/∂a_h1 can be represented as below with subsequent steps:
        - ∂E1/∂a_h1 = ∂E1/∂a_o1\*∂a_o1/∂o1\*∂o1/∂a_h1
        - ∂E1/∂a_h1 = (a_o1 - t1) \* a_o1 \* (1 - a_o1) \* w5
      - Similarly ∂E2/∂a_h1 can be written as:
        - ∂E2/∂a_h1 = (a_o2 - t2) \* a_o2 \* (1 - a_o2) \* w7 
      - ∂E_total/∂a_h1 becomes:
        - ∂E_total/∂a_h1 = (a_o1 - t1) \* a_o1 \* (1 - a_o1) \* w5 + (a_o2 - t2) \* a_o2 \* (1 - a_o2) \* w7
        
4. Similar calculation is done for ∂E_total/∂a_h2:
      - ∂E_total/∂a_h2 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8

5. Gradient for w1 is represented as below:
      - ∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1
      - ∂E_total/∂a_h1 is already calculated in step 3
      - ∂a_h1/∂h1 is sigmoid differential
      - ∂h1/∂w1 is equal to i1
      - Overall equation for ∂E_total/∂w1 can be represented as:
        - ∂E_total/∂w1 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * ( a_h1 * (1 - a_h1) ) * i1

6. Similar calculations can be performed on w2, w3 and w4. Following are their gradient respectively
      - ∂E_total/∂w2 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * ( a_h1 * (1 - a_h1) ) * i2
      - ∂E_total/∂w3 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * ( a_h2 * (1 - a_h2) ) * i1
      - ∂E_total/∂w4 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * ( a_h2 * (1 - a_h2) ) * i2

7. Once we have gradient values for all, feedforward starts. Weights are adjusted using these gradients after multiplying them with Learning rate.
8. This process is repeated a number of times, we say it as epoch.

In the current example we are working we have assumed values of input values, output values & initial weights. By changing the number of learning rates we are trying to understand the nature of convergence.

We are changing the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] and analysing it via loss graph.

Learning rate 0.1:

![image](https://github.com/MPGarg/ERA1_Session6/assets/120099863/7072819b-5421-4888-a2f7-8e0949303c84)

Learning rate 0.2:

![image](https://github.com/MPGarg/ERA1_Session6/assets/120099863/80555086-d21f-42ec-8083-dba4edf593a8)

Learning rate 0.5:

![image](https://github.com/MPGarg/ERA1_Session6/assets/120099863/8195c0b3-cdaa-411a-bfc6-396c50c89463)

Learning rate 0.8:

![image](https://github.com/MPGarg/ERA1_Session6/assets/120099863/3a616d01-ecea-47c7-a672-3c24794c0299)

Learning rate 1.0:

![image](https://github.com/MPGarg/ERA1_Session6/assets/120099863/4a4ba3b9-ba64-4ac6-8abd-d44596ea6fbe)

Learning rate 2.0:

![image](https://github.com/MPGarg/ERA1_Session6/assets/120099863/eb5854f6-ab72-4681-b736-6fd0036f011f)

It can be seen that for this dataset, as we are increasing learning rate loss is getting reduced rapidly. But if we keep on increasing it, loss might not converge at all!

## Assignment Session-6 Part-2

Problem statement:
Design network fot MNIST which satisfies following conditions:

![image](https://github.com/MPGarg/ERA1_Session6/assets/120099863/898822d2-27cb-4cd3-920c-e4034ae361d5)

Following components are used in designing of netwrok:
- 3x3 Convolutions
- 1x1 Convolutions
- MaxPooling
- Batch Normalization
- Image Normalization
- DropOut
- Fully Connected Layer
- GAP

The summary of network designed:

![image](https://user-images.githubusercontent.com/120099863/211984618-5e04cdd4-a5bf-4699-9317-480f13a7faf2.png)

Major points about Network designed:
- Network has total 9 layers 
- Logic used for designing layers is CRB (Convolution-Relu-Batch Normalization)
- Dropout of 0.05% is used after Batch Normalization layer
- Dropout & Batch Normalization not used after 1x1 convolution layer
- 1x1 convolution is used after two 3x3 convolutions followed by Max pooling
- Number of channels vary from 8 to 32 at different layers
- GAP is used near to last layer after convolution and a layer before fully connected layer
- Fully connected layer is last layer of network
- Log Softmax used as last layer activation function with NLL Loss 

Trainable parameters for network are 16,562 (less than 20k) and it was able to achieve 99.4% test/validation accuracy consistently from 16th epoch onwards. Network reached maximum validation accuracy at 18th epoch to 99.5% (9946/10000). 

![image](https://user-images.githubusercontent.com/120099863/211985645-86cb54fd-daca-4b25-9461-35d2b130dd6f.png)
