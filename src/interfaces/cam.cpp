#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    torch::jit::script::Module model;

    const std::string PATH = "src/interfaces/best_model.pth";
    try {
        model = torch::jit::load(PATH); // Carrega o modelo treinado
        model.to(torch::kCPU);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Erro ao carregar o modelo: " << e.what() << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir a câmera!" << std::endl;
        return -1;
    }

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::resize(frame, frame, cv::Size(224, 224));
        frame.convertTo(frame, CV_32F, 1.0f / 255.0f);

        torch::Tensor input = torch::from_blob(frame.data, {1, 224, 224, 3}, torch::kFloat).clone();
        input = input.permute({0, 3, 1, 2}).contiguous();

        // Inferência com o modelo
        try {
            torch::Tensor output = model.forward({input}).toTensor();
            std::cout << "Inferência realizada com sucesso!" << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Erro na inferência: " << e.what() << std::endl;
        }

        cv::imshow("Câmera + Inferência", frame);
        if (cv::waitKey(1) == 27) break; // ESC para sair
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
