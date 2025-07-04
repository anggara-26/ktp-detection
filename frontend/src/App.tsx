import { useState } from "react";

const API_URL = "https://dd38-103-47-133-130.ngrok-free.app";

function App() {
  const [image, setImage] = useState<File | null>(null);
  const [response, setResponse] = useState<{
    extracted_data: {
      nik: string;
      nama: string;
      ttl: string;
      jeniskelamin: string;
      alamat: string;
      rtrw: string;
      keldesa: string;
      kecamatan: string;
      agama: string;
      statuskawin: string;
      pekerjaan: string;
      kewarganegaraan: string;
    };
    foto_image: string;
    ktp_image: string;
    list_of_images: string[];
  } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const treshold = formData.get("treshold") as string;
    if (!image) {
      alert("Please select a file");
      return;
    }

    formData.append("image_file", image);
    formData.append("confidence_threshold", "0.5");

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 120s timeout (2 minutes)

    try {
      setIsLoading(true);
      const response = await fetch(`${API_URL}/api/ocr/`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
        // Do NOT set headers here; browser will set correct Content-Type and boundary
      });
      clearTimeout(timeoutId);

      const responseJson = await response.json();

      if (!response.ok) {
        throw new Error(responseJson.detail || "Something went wrong");
      }

      setResponse(responseJson);
      console.log("Response:", responseJson);
    } catch (error: any) {
      if (error.name === "AbortError") {
        alert("Request timed out. The server may be slow or unresponsive.");
      } else if (error.message && error.message.includes("Failed to fetch")) {
        alert(
          "Network error: Could not connect to the server. Please check your connection or server status."
        );
      } else {
        alert("Error: " + (error.message || error));
      }
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }
  };
  return (
    <form
      onSubmit={onSubmit}
      style={{
        display: "flex",
        gap: "1rem",
        flexDirection: "column",
        margin: "0 auto",
        width: "100vw",
        minHeight: "100vh",
        justifyContent: "center",
        overflow: "auto",
        overflowX: "hidden",
        // alignItems: "center",
      }}
    >
      <div
        style={{
          display: "flex",
          gap: "1rem",
          flexDirection: "column",
          maxWidth: "400px",
          margin: "0 auto",
        }}
      >
        <input
          type="file"
          accept="image/*"
          name="file"
          onChange={(e) => {
            const files = e.target.files;
            if (files && files.length > 0) {
              setImage(files[0]);
            }
          }}
        />
        <input type="number" name="treshold" />
        <button type="submit" disabled={isLoading}>
          Submit
        </button>
        <div style={{ display: "flex", gap: "1rem", marginTop: "1rem" }}>
          {isLoading && <p>Loading...</p>}
          {response && !isLoading && (
            <div>
              <h3>Extracted Data:</h3>
              <ul>
                {Object.entries(response.extracted_data).map(([key, value]) => (
                  <li key={key}>
                    {key}: {value}
                  </li>
                ))}
              </ul>
              <img
                src={`data:image/jpeg;base64,${response.foto_image}`}
                alt="Foto"
                style={{ maxWidth: "200px", maxHeight: "200px" }}
              />
              <img
                src={`data:image/jpeg;base64,${response.ktp_image}`}
                alt="KTP"
                style={{ maxWidth: "200px", maxHeight: "200px" }}
              />
              <h3>List of Images:</h3>
              {response.list_of_images.map((image, index) => (
                <img
                  key={index}
                  src={`data:image/jpeg;base64,${image}`}
                  alt={`Image ${index}`}
                  style={{ maxWidth: "200px", maxHeight: "200px" }}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </form>
  );
}

export default App;
