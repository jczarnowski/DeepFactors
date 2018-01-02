#include <memory>
#include <iostream>

#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session.h>

int main(int argc, char** argv)
{
    tensorflow::port::InitMain(argv[0], &argc, &argv);
    
    std::unique_ptr<tensorflow::Session> session;
    tensorflow::SessionOptions sess_opt;
    sess_opt.config.mutable_gpu_options()->set_allow_growth(true);
    (&session)->reset(tensorflow::NewSession(sess_opt));
    
    if (!session->Create(graph_def).ok()) 
    {
        std::cout << "Create graph";
        return -1;
    }
}
