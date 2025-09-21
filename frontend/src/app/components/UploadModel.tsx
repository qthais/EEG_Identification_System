"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { FaTimes, FaUpload } from "react-icons/fa";

interface UploadModalProps {
  onClose: () => void;
  onFileSelect: (file: File) => void;
}

const UploadModal: React.FC<UploadModalProps> = ({ onClose, onFileSelect }) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
      onClose();
    }
  }, [onClose, onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/octet-stream": [".edf"] }, // chỉ cho phép .edf
    multiple: false,
  });

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-lg p-6 w-full max-w-md relative">
        <button
          onClick={onClose}
          className="absolute top-2 right-2 text-gray-500 hover:text-black"
        >
          <FaTimes />
        </button>

        <h2 className="text-lg font-semibold mb-4">Upload EEG File</h2>

        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer ${
            isDragActive ? "border-pink-500 bg-pink-50" : "border-gray-300"
          }`}
        >
          <input {...getInputProps()} />
          <FaUpload className="mx-auto mb-3 text-3xl text-gray-500" />
          {isDragActive ? (
            <p className="text-pink-600">Drop your .edf file here...</p>
          ) : (
            <p className="text-gray-600">
              Drag & drop your .edf file here, or <span className="text-pink-600">click to select</span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default UploadModal;
