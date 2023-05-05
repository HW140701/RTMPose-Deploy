#ifndef _ONNXRUNTIME_CPU_MODEL_BASE_H_
#define _ONNXRUNTIME_CPU_MODEL_BASE_H_

#include <string>
#include <thread>

#include "opencv2/opencv.hpp"

#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"

class OnnxruntimeCPUModelBase
{
public:
	OnnxruntimeCPUModelBase();
	virtual~OnnxruntimeCPUModelBase();

public:
	virtual bool LoadModel(const std::string& onnx_env_name, const std::string& onnx_model_path);

protected:
	virtual void PrintModelInfo(Ort::Session& session);

protected:
	Ort::Env m_env;
	Ort::Session m_session;
};

#endif // !_ONNXRUNTIME_CPU_MODEL_BASE_H_
