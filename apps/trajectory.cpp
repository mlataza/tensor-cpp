#include <tensor.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace tc;

std::ostream &print(std::ostream &os, double d)
{
    return os << std::setw(12) << std::fixed << std::setprecision(6) << d;
}

std::ostream &operator<<(std::ostream &os, const tensor<4> &t)
{
    return print(print(print(print(os, t(0)) << ", ", t(1)) << ", ", t(2)) << ", ", t(3));
}

tensor<4> initial_state(double s, double angle)
{
    return tensor<4>{0.0, 0.0, s * std::cos(angle), s * std::sin(angle)};
}

tensor<4> x_dot(const tensor<4> &x)
{
    static const tensor<4, 4> A{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    static const tensor<4> B{0.0, 0.0, 0.0, -9.81};

    return A * x + B;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: trajectory <speed> <angle>\n";
        return 1;
    }

    auto dt = 0.000001;
    auto Dt = 0.01;
    auto steps = (int)std::floor(Dt / dt);

    // Initialize variables
    auto s0 = std::stod(argv[1]);
    auto angle0 = std::stod(argv[2]) * M_PI / 180.0;
    auto x = initial_state(s0, angle0);
    auto t = 0.0;
    auto h = 0.0;

    std::cout << "---------------------------------------------------------------------\n";
    std::cout << "           t |                       r(t),                       v(t)\n";
    std::cout << "---------------------------------------------------------------------\n";

    for (auto i = 0; x(1) >= 0.0; t += dt, ++i)
    {
        x += x_dot(x) * dt;
        h = std::max(h, x(1));

        if (i % steps == 0)
        {
            print(std::cout, t) << " | " << x << '\n';
        }
    }
    print(std::cout, t) << " | " << x << '\n';

    std::cout << "---------------------------------------------------------------------\n";
    print(std::cout << "        time | ", t) << " s\n";
    print(std::cout << "       range | ", x(0)) << " m\n";
    print(std::cout << "      height | ", h) << " m\n";
    std::cout << "---------------------------------------------------------------------\n";

    return 0;
}