#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap("http://192.168.1.3:4747/video", cv::CAP_FFMPEG); // 0 para webcam, ou ip

    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir a câmera!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;  // Captura um frame da câmera

        if (frame.empty()) break;

        cv::imshow("Câmera ao Vivo", frame);

        if (cv::waitKey(1) == 27) break; // Pressione ESC para sair
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
