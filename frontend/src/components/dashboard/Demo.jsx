import { useState } from "react";
import Section from "../Section";
import Button from "../Button";
import VideoStream from "../VideoStream";

const Demo = () => {
  const [analysisResult, setAnalysisResult] = useState("");
  const [personName, setPersonName] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [finalResult, setFinalResult] = useState([]);
  const [btnExtract, setBtnExtract] = useState(true);
  const [btnTest, setBtnTest] = useState(true);
  const [eventSource, setEventSource] = useState(null);

  // const videoStreamURL = "http://192.168.1.37:8080/video";
  const videoStreamURL = "http://192.168.43.1:8080/video";

  const restartServer = async () => {
    setBtnExtract(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/restart", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const data = await response.json();
      if (response.ok) {
        alert(data.message);
      } else {
        alert(data.error || "An error occurred while restarting the server.");
      }
    } catch (error) {
      console.error("Server Restart Error:", error);
      alert("An error occurred while restarting the server. Please try again.");
    }
  };

  const handleExtract = async () => {
    if (!personName) {
      alert("Please enter a person's name before training.");
      return;
    }
    setBtnExtract(false);
    setIsProcessing(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          person_name: personName,
          video_stream_url: videoStreamURL,
        }),
      });
      const data = await response.json();
      alert(data.message || data.error);
    } catch (error) {
      console.error("Training Error:", error);
      alert("An error occurred during training. Please try again.");
    }
  };

  const handleTest = () => {
    setBtnTest(false);
    const newEventSource = new EventSource(
      `http://127.0.0.1:5000/predict?video_stream_url=${encodeURIComponent(videoStreamURL)}`
    );
    setEventSource(newEventSource);

    setAnalysisResult("Waiting for predictions...");
    setIsProcessing(true);

    newEventSource.onmessage = (event) => {
      setFinalResult((prevItems) => [...prevItems, event.data]);
      setAnalysisResult(event.data);
    };

    newEventSource.onerror = (error) => {
      console.error("SSE Error:", error);
      alert("An error occurred during testing.");
      newEventSource.close();
    };

    // Stop the stream after 30 seconds
    setTimeout(() => {
      setIsProcessing(false);
      newEventSource.close();
    }, 30000);
  };

  const handleTestStop = () => {
    if (eventSource) {
      eventSource.close();
      setEventSource(null);
    }

    if (finalResult.length > 0) {
      const frequency = {};
      finalResult.forEach((item) => {
        frequency[item] = (frequency[item] || 0) + 1;
      });

      const mostCommonPrediction = Object.keys(frequency).reduce((a, b) =>
        frequency[a] > frequency[b] ? a : b
      );
      setAnalysisResult(`Person is ${mostCommonPrediction}`);
    } else {
      setAnalysisResult("No predictions were made.");
    }

    setFinalResult([]);
    setBtnTest(true);
    restartServer()
  };

  const handleTrain = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/train", {
        method: "POST",
      });
      const data = await response.json();
      alert(data.message || data.error);
    } catch (error) {
      console.error("Training Error:", error);
      alert("An error occurred during training. Please try again.");
    }
  };

  const stopProcess = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await response.json();
      alert(data.message || data.error);
      setBtnExtract(true);
    } catch (error) {
      console.error("Stop Error:", error);
      alert("An error occurred while stopping the process. Please try again.");
    }
  };
  

  return (
    <div className="bg-gray-100 min-h-screen">
      <div className="grid grid-cols-2 gap-20 px-10 py-6 h-[110vh]">
        <div className="space-y-6 border">
          <h1 className="text-xl font-bold text-gray-700 text-center">Extracting</h1>
          <VideoStream streamURL={videoStreamURL} />
          <div className="px-10 py-6">
            <label className="block text-lg font-medium text-gray-700">
              Enter Person Name:
            </label>
            <input
              type="text"
              value={personName}
              onChange={(e) => setPersonName(e.target.value)}
              className="w-full p-2 border border-gray-300"
              disabled={isProcessing}
            />
          </div>
          <div className="flex justify-end space-x-4">
            {btnExtract ? (
              <Button
                label={"Start Extracting"}
                variant="success"
                onClick={handleExtract}
                disabled={isProcessing}
              />
            ) : (
              <Button label={"Stop"} variant="danger" onClick={stopProcess} />
            )}
            <Button label={"Train"} variant="primary" onClick={handleTrain} />
          </div>
        </div>
        <div className="space-y-6 border">
          <h1 className="text-xl font-bold text-center text-gray-700">Testing</h1>
          <VideoStream streamURL={videoStreamURL} />
          <div className="flex justify-end space-x-4">
            {btnTest ? (
              <Button
                label={"Start Testing"}
                variant="success"
                onClick={handleTest}
                disabled={isProcessing}
              />
            ) : (
              <Button label={"Stop"} variant="danger" onClick={handleTestStop} />
            )}
          </div>
        </div>
      </div>
      <div className="px-10 py-6">
        <Section
          id="analysis"
          title="Analysis Output"
          description={analysisResult || "Waiting for predictions..."}
          height="h-40"
        />
      </div>
    </div>
  );
};

export default Demo;
