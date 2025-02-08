#include <torch/script.h> // LibTorch
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    torch::jit::script::Module model;

    const std::string PATH = "src/saved/classification.pth";
    model = torch::jit::load(PATH); // Carrega o modelo treinado

    cv::Mat frame;
    cv::VideoCapture cap(0);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Pré-processamento da imagem
        cv::resize(frame, frame, cv::Size(224, 224));
        torch::Tensor input = torch::from_blob(frame.data, {1, 224, 224, 3}, torch::kFloat);
        input = input.permute({0, 3, 1, 2}); // Reordenar dimensões para PyTorch

        // Inferência com o modelo
        torch::Tensor output = model.forward({input}).toTensor();

        cv::imshow("Câmera + Inferência", frame);
        if (cv::waitKey(1) == 27) break; // ESC para sair
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
