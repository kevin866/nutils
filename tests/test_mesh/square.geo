Point(1) = {0,0,0};
Point(2) = {1,0,0};
Point(3) = {1,1,0};
Point(4) = {0,1,0};
Line(5) = {1,2};
Line(6) = {2,3};
Line(7) = {3,4};
Line(8) = {4,1};
Line Loop(9) = {5,6,7,8};
Plane Surface(10) = {9};
Physical Point("corner") = {2};
Physical Line("neumann") = {5};
Physical Line("dirichlet") = {6,7,8};
Physical Surface("interior") = {10};
