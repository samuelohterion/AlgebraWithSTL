# AlgebraWithSTL
algebra, some math with std::vector&lt; T > and std::vector&lt; std::vector&lt; T > > 
This is a small collection of some functions and operator overloadings for convenient work
with STL std::vector< T > as a mathematical vector and std::vector< std::vector< T > > as a mathematical matrix.

## creating vectors and matrices

### creating vectors
    std::vector< double >
    u = { 1., 2. };
    // or shorter Vec< double > 
    // or even shorter but only for double available
    VD
    v = { -1., 0., +1. };

### creating matrices
    std::vector< std::vector< double > >
    a = {
     {  1., 2. },
     { -2., 1. } };
     // or shorter Mat< double >
     // or even shorter but only for double available
    MD
    b = {
     { 1.,  0.,  0. },
     { 0., +3., +4. },
     { 0., -4., +3. } };
     
## unary operators
### +/-
    std::cout << +u << std::endl;
    std::cout << -v << std::endl;
    
    std::cout << +a << std::endl;
    std::cout << -b << std::endl;

### ~ (transpose of a matrix)
    std::cout << ~b << std::endl;
    
