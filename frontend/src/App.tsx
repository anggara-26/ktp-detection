import { useState } from "react";

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

    try {
      setIsLoading(true);
      const response = await fetch("http://127.0.0.1:8000/api/ocr/", {
        method: "POST",
        body: formData,
        // headers: {
        //   "Content-Type": "application/form-data",
        // },
      });

      const responseJson = await response.json();

      if (!response.ok) {
        throw new Error(responseJson.detail || "Something went wrong");
      }

      setResponse(responseJson);
      console.log("Response:", responseJson);
    } catch (error) {
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
        <button type="submit">Submit</button>
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
