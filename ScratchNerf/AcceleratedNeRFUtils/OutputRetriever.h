#pragma once
#include <cstdint>

namespace AcceleratedNeRFUtils {
	using namespace System;
	using namespace System::Numerics;
	public ref class OutputRetriever
	{
	public:
		static array <Vector3>^ RetrieveOutput(uint64_t dev_output, int size);
	};
}

