# PolaTrace, a polarization mathematic simulation Python library 
Author: Xiaojun(James) Chen from Inline Photonics Inc.  Email: jchen@inlinephotonics.com

This open-source Python program was developed  for the book  "Polarization Measurement and Control in Optical Fiber Communication and Sensor Systems."  By X. Steve Yao and Xiaojun(James) Chen. The package includes the following:

1) PolaTrace library named polatrace.py is a library to display and emulates the optical polarization phenomenon in fiber.  

2) Application examples of the PolarTrace library:

Example 1: PolarizationEllipse.py. It displays polarization ellispe according to the inputs of  amplitudes of Ex and Ey,  and the phase difference between Ey and Ex: 
         ![image](https://user-images.githubusercontent.com/110875419/202661174-5ece0e1a-71b0-4345-b314-1d9f63548426.png)

Example 2: JV-Ellipse-Stokes.py. It represents the polarization of a given optical electric field in polarization ellispe, and a Stokes vector on Poincare sphere. 
![image](https://user-images.githubusercontent.com/110875419/202660814-0861b550-6973-4e29-972a-c6690dd78fbe.png)

Example 3: Polarizaion elements.py.  It displays the relationship between the input and output polarization state after light passes a polarization element, for example, waveplate, rotator, and partial polarizer. This program also shows the trace output polarization on the Poincare sphere when retardation or optical axis orientation is changed. This program can be used as a conversation between the Jones vector and the Stokes parameters of monochromatic light.  
![image](https://user-images.githubusercontent.com/110875419/202667384-e21b8ebd-5738-4ddf-b7cc-3dcff0bd3fdf.png)

Example 4: Spun fiber.py. This program calculates and displays the polarization states evaluation along a spun fiber. The spun fiber is modeled by a continuously rotated linear polarization maintaining fiber. One can change the input polarization state, intrinsic linear birefringence, and rotating rate to see how these parameters influence polarization evaluation along the spun fiber.  
![image](https://user-images.githubusercontent.com/110875419/202670038-d6bdb92e-0b0f-48e6-bd63-6e5516a1b850.png)
