import { useState } from "react";
import Header from "../Model2/header";

const Recodered = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

const handleSubmit = async (e) => {
  e.preventDefault();

  if (!selectedFile) {
    alert("Please upload a video file.");
    return;
  }

  const formData = new FormData();
  formData.append("video", selectedFile);

  setLoading(true);
  setPrediction("");

  try {
    const response = await fetch("http://localhost:5000/r_predict", {
      method: "POST",
      body: formData,
    });

    console.log("Response status:", response.status);

    if (!response.ok) {
      const errorData = await response.json();
      console.error("Error response:", errorData);
      throw new Error(errorData.error || "Failed to get a response from the backend.");
    }
    
    const data = await response.json();
    console.log("Predicted data:", data);
    setPrediction(data.predicted_person || "Unknown");
  } catch (error) {
    console.error("Error:", error);
    setPrediction("Error predicting the person.");
  } finally {
    setLoading(false);
  }
};


  const handleTrain = async () => {
    setTraining(true);

    try {
      const response = await fetch("http://localhost:5000/r_train", {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("Failed to train the model.");
      }

      alert("Model training completed successfully.");
    } catch (error) {
      console.error("Error during training:", error);
      alert("Error during model training.");
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Header />
      <div className="container mx-auto py-10">
        
        <form
          onSubmit={handleSubmit}
          className="max-w-md mx-auto bg-white p-6 rounded-lg shadow-md"
        >
          <label className="block text-sm font-medium text-gray-700">
            Upload Video File
          </label>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          />
          <button
            type="submit"
            className="mt-4 w-full bg-blue-500 text-white py-2 px-4 rounded-md shadow hover:bg-blue-600 transition duration-300"
            disabled={loading || training}
          >
            {loading ? "Predicting..." : "Submit"}
          </button>
        </form>

        <div className="flex justify-center">
          <button
            onClick={handleTrain}
            className="mt-4 w-96 bg-green-500 text-white py-2 px-4 rounded-md shadow hover:bg-green-600 transition duration-300"
            disabled={training || loading}
          >
            {training ? "Training..." : "Train Model"}
          </button>
        </div>

        {prediction && (
          <div className="max-w-md mx-auto mt-6 bg-green-100 border-l-4 border-green-500 text-green-700 p-4 rounded">
            <p className="font-bold">Prediction Result:</p>
            <p>{prediction}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Recodered;
