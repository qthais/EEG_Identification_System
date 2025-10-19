"use client";

import { FaMicrophone, FaUpload } from "react-icons/fa";
import Button from "../components/Button";
import NavBar from "../components/NavBar";
import Footer from "../components/Footer";
import Image from "next/image";
import { useState } from "react";
import axios from "axios";
import toast from "react-hot-toast";
import UploadModal from "../components/UploadModel";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";
const apiUrl = process.env.NEXT_PUBLIC_API_URL;

export default function SignInPage() {
    const [file, setFile] = useState<File | null>(null);
    const [showModal, setShowModal] = useState(false);
    const [isSignIn, setIsSignIn] = useState(false);
    const router = useRouter();
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsSignIn(true);
        if (!file) {
            toast.error("Please select an EDF file");
            setIsSignIn(false);
            return;
        }

        const formData = new FormData();
        formData.append("file", file); // ✅ backend yêu cầu key = file

        try {
            const res = await axios.post(`${apiUrl}/api/login_eeg`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            if (res.data.success) {
                const token = res.data.data.access_token;
                toast.success(
                    `✅ Login OK! User class: ${res.data.data.predicted_class} (conf: ${(
                        res.data.data.confidence * 100
                    ).toFixed(2)}%)`
                );
                const result = await signIn("credentials", {
                    redirect: false,
                    accessToken: token,
                    callbackUrl: "/dashboard",
                });

                if (result?.ok && !result.error) {
                    router.push(result.url || "/dashboard");
                } else {
                    toast.error(`❌ NextAuth login failed`);
                }
            } else {
                toast.error(`❌ ${res.data.message || "Login failed"}`);
            }
            setIsSignIn(false);
        } catch (err: any) {
            setIsSignIn(false);
            if (err.response) {
                toast.error(`❌ ${err.response.data.message || "Server error"}`);
            } else {
                toast.error("Network error");
            }
        }
    };
    return (
        <>
            <NavBar />
            <main className="relative min-h-screen flex flex-col bg-transparent">
                <Image
                    src="/background.png"
                    alt="Background"
                    fill
                    className="object-cover -z-10"
                    priority
                />
                {/* Content */}
                <div className="flex flex-1 items-center justify-center px-4">
                    <div className="bg-white shadow-2xl rounded-2xl p-10 w-full max-w-md text-center">
                        <h1 className="text-xl font-semibold">Welcome to NeuroID</h1>
                        <p className="text-gray-500 text-sm mt-2">
                            Quick & Secure way to know who you are
                        </p>

                        {/* Buttons */}
                        <div className="mt-8 flex flex-col gap-4">
                            <Button className="flex items-center justify-center gap-2 bg-black text-white w-full py-3">
                                <FaMicrophone /> Real time EEG Login
                            </Button>

                            <span className="text-gray-400 text-xs">Or</span>

                            <form
                                onSubmit={handleSubmit}
                                className="flex flex-col items-center gap-3 w-full"
                            >
                                <Button
                                    outline
                                    type="button"
                                    className="flex items-center justify-center gap-2 w-full py-3"
                                    onClick={() => setShowModal(true)}
                                >
                                    <FaUpload /> Upload EEG data to login
                                </Button>

                                {file && (
                                    <p className="text-sm text-gray-500">
                                        Selected: <span className="font-medium">{file.name}</span>
                                    </p>
                                )}

                                <Button type="submit" className="bg-pink-600 text-white w-full py-3" disabled={isSignIn}>
                                    {isSignIn ? "Logging..." : "Login"}
                                </Button>
                            </form>
                        </div>
                    </div>
                </div>

                <Footer />
                {/* Popup modal */}
                {showModal && (
                    <UploadModal
                        onClose={() => setShowModal(false)}
                        onFileSelect={(f) => setFile(f)}
                    />
                )}
            </main>
        </>
    );
}