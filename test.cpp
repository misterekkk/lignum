#include "lignum.hpp"

#include <iostream>
#include <vector>
#include <exception>

int main() {
    try {
        std::cout << "Ladowanie JSONa\n";
        lignum::Model model_json = lignum::load("model_xgb.json", "xgboost");
        std::cout << "Zaladowano JSONa\n";

        std::cout << "Test zapisu do binarki\n";
        model_json.save("model.bin");
        std::cout << "Zapisano do model.bin\n";

        std::cout << "Test odczytu binarki\n";
        lignum::Model model_bin = lignum::load("model.bin");
        std::cout << "Zaladowano binarke\n";

        std::cout << "Test predykcji\n";
        std::vector<double> X(20, 0.0); 
        std::vector<double> preds(1, 0.0);

        model_bin.predict(X.data(), 1, 20, preds.data());

        std::cout << "Predykcja dla wektora zer: " << preds[0] << "\n";
        std::cout << "Sukces\n";

    } catch (const std::exception& e) {
        std::cerr << "Blad\n" << e.what() << "\n";
        return 1;
    }

    return 0;
}