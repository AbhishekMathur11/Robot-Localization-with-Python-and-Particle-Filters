# Robot-Localization-with-Python-and-Particle-Filters
When a UGV, drone, or any other robot is sent to an unknown place, its terrain is always unknown and undulating. But using a very ingenious method called Particle Filters, makes the whole process of localizing the terrain and understanding its elevation and depression quite simple. It conveniently incorporates Artificial Intelligence to remember the terrain which the robot traveled upon. Using python, I worked in conjunction with the Coursera instructor and developed this method in this project.
Particle Filters is a method which involves a physical map upon which numerous particles are scattered and each particle has a certain weight value pertaining to the elevation of that point in the terrain.
Using these particles, any robot in that space can navigate with the chances of occurence of noise, which gets minimised by more number of particles.
Mathematically, we can use probability of noise occurence in the form of normal Gaussian distribution ans the noise in this case represents the standard deviation against the displacement that is provided through user input
More the number of particle filters, more accurate will the controllability of the robot be.
This can be used in all types of robots, including droids that are sent to other planets that interact with unknown exosystems and map their terrain. Pretty useful!
