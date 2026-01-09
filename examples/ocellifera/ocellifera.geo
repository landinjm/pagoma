// Okay we're drawing a Costasiella ocellifera in 3D!
SetFactory("OpenCASCADE");

// Characteristic length
lc = 5e-1;

// Parameters
under_major_axis = 3;
under_minor_axis = 1;
height = 1;
num_layers = 4;
taper = 0.4;

// Create ellipse curves at different heights
For i In {0:num_layers}
  z = height * i / num_layers;
  scale = 1 - taper * (z / height);

  // Create center point
  Point(100 + i*5 + 1) = {0, 0, z};

  // Create points on major and minor axes
  Point(100 + i*5 + 2) = {under_major_axis * scale, 0, z};
  Point(100 + i*5 + 3) = {0, under_minor_axis * scale, z};
  Point(100 + i*5 + 4) = {-under_major_axis * scale, 0, z};
  Point(100 + i*5 + 5) = {0, -under_minor_axis * scale, z};

  // Create ellipse arcs
  Ellipse(100 + i*4 + 1) = {100 + i*5 + 2, 100 + i*5 + 1, 100 + i*5 + 2, 100 + i*5 + 3};
  Ellipse(100 + i*4 + 2) = {100 + i*5 + 3, 100 + i*5 + 1, 100 + i*5 + 3, 100 + i*5 + 4};
  Ellipse(100 + i*4 + 3) = {100 + i*5 + 4, 100 + i*5 + 1, 100 + i*5 + 4, 100 + i*5 + 5};
  Ellipse(100 + i*4 + 4) = {100 + i*5 + 5, 100 + i*5 + 1, 100 + i*5 + 5, 100 + i*5 + 2};

  // Create wire (curve loop)
  Wire(100 + i) = {100 + i*4 + 1, 100 + i*4 + 2, 100 + i*4 + 3, 100 + i*4 + 4};
EndFor

// Create wires array for ThruSections
body_wires[] = {};
For i In {0:num_layers}
  body_wires[] += {100 + i};
EndFor

// ThruSections creates a solid through the wires
ThruSections(1) = {body_wires[]};

// Add sphere for head
head_radius = 0.7;
head_x = under_major_axis * 0.8;
head_y = 0;
head_z = height * 0.5;

Sphere(2) = {head_x, head_y, head_z, head_radius};

// Create a box below z=0 to cut off the sphere
box_size = 10;  // Large enough to contain the sphere
Box(3) = {head_x - box_size, head_y - box_size, -box_size,
          2*box_size, 2*box_size, box_size};  // From z=-box_size to z=0

// Cut the sphere with the box (remove everything below z=0)
BooleanDifference{Volume{2}; Delete;}{Volume{3}; Delete;}

// Merge body and head
BooleanUnion{Volume{1}; Delete;}{Volume{2}; Delete;}

// Set the characteristic length to everything we've defined so far
Mesh.CharacteristicLengthFactor = lc;
