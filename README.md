# Robot Localization with Python and Particle Filters

When a UGV, drone, or any other robot is sent to an unknown place, its terrain is always unknown and undulating. But using a very ingenious method called Particle Filters, makes the whole process of localizing the terrain and understanding its elevation and depression quite simple. It conveniently incorporates Artificial Intelligence to remember the terrain which the robot traveled upon. 

This project implements the Particle Filters method using Python in collaboration with a Coursera instructor. Particle Filters is a method which involves a physical map upon which numerous particles are scattered. Each particle has a certain weight value pertaining to the elevation of that point in the terrain. 

Using these particles, any robot in that space can navigate with the chances of occurrence of noise, which gets minimized by a larger number of particles. Mathematically, we can use the probability of noise occurrence in the form of a normal Gaussian distribution, where the noise represents the standard deviation against the displacement provided through user input. 

More the number of particle filters, more accurate will the controllability of the robot be. This method can be used in all types of robots, including droids that are sent to other planets to interact with unknown exosystems and map their terrain. It is incredibly useful!

## Project Guide

To get started with the project, follow the steps below:

1. **Clone the repository**:
   - Clone this repository to your local machine using the following command:
     ```
     git clone https://github.com/your-username/Robot-Localization-with-Python-and-Particle-Filters.git
     ```

2. **Navigate to the project directory**:
   - Change your current directory to the project directory:
     ```
     cd Robot-Localization-with-Python-and-Particle-Filters
     ```

3. **Run the Particle Filters control file**:
   - Execute the Python file that controls the Particle Filters:
     ```
     python particle_filters.py
     ```

4. **Access the terrain map**:
   - To view the terrain map, click on the following link:
     - [Terrain Map](https://github.com/IronAvenger11-prog/Robot-Localization-with-Python-and-Particle-Filters/blob/main/map.png)

## Additional Resources

For a deeper understanding of Particle Filters and its implementation, you can refer to the following resources:

- [Particle Filters - Wikipedia](https://en.wikipedia.org/wiki/Particle_filter)
- [Particle Filters Tutorial - Coursera](https://www.coursera.org/learn/robotics-perception)






