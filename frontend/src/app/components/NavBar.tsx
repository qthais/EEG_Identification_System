"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import Container from "./Container";
import { User } from "@/types/User";
interface NavBarProps {
  user?: User | null;
}


const NavBar: React.FC<NavBarProps> = ({ user }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Container>
      <nav className="w-full flex items-center justify-between my-3 h-full">
        {/* Logo */}
        <Link href={'/'} className="flex items-center space-x-2 cursor-pointer">
          <Image src="/Logo.png" alt="NeuroID" height={40} width={150} />
        </Link>

        {/* Desktop Menu */}
        <div className="hidden sm:flex items-center space-x-6">
          <Link href="/features" className="text-sm font-semibold hover:text-pink-600">
            Features
          </Link>
          <Link href="/support" className="text-sm font-semibold hover:text-pink-600">
            Support
          </Link>
          <Link href="/about" className="text-sm font-semibold hover:text-pink-600">
            About
          </Link>
          {user ? (
            <button
              onClick={() => alert("Sign out logic here")}
              className="ml-2 px-4 py-2 bg-black text-white text-sm font-semibold rounded-full hover:bg-pink-700"
            >
              Sign Out
            </button>
          ) : (
            <>
              <Link href="/signin" className="text-sm font-semibold hover:text-pink-600">
                Sign In
              </Link>
              <Link
                href="/signup"
                className="ml-2 px-4 py-2 bg-black text-white text-sm font-semibold rounded-full hover:bg-pink-700"
              >
                Sign Up
              </Link>
            </>
          )}
        </div>

        {/* Mobile Hamburger */}
        <button
          className="sm:hidden flex flex-col space-y-1"
          onClick={() => setIsOpen(!isOpen)}
        >
          <span className="w-6 h-0.5 bg-black"></span>
          <span className="w-6 h-0.5 bg-black"></span>
          <span className="w-6 h-0.5 bg-black"></span>
        </button>
      </nav>

      {/* Mobile Menu */}
      {isOpen && (
        <div className="md:hidden min-w-[200px] fixed top-14 right-4 z-50 flex flex-col items-center bg-white shadow-lg rounded-lg px-6 py-4 space-y-4">
          <Link
            href="/features"
            className="w-full text-center py-2 text-sm font-semibold hover:bg-gray-100 rounded transition"
          >
            Features
          </Link>
          <Link
            href="/support"
            className="w-full text-center py-2 text-sm font-semibold hover:bg-gray-100 rounded transition"
          >
            Support
          </Link>
          <Link
            href="/about"
            className="w-full text-center py-2 text-sm font-semibold hover:bg-gray-100 rounded transition"
          >
            About
          </Link>
          <Link
            href="/signin"
            className="w-full text-center py-2 text-sm font-semibold hover:bg-gray-100 rounded transition"
          >
            Sign In
          </Link>
          <Link
            href="/signup"
            className="w-full text-center py-2 text-sm font-semibold bg-black text-white rounded-full hover:bg-pink-700 transition"
          >
            Sign Up
          </Link>
        </div>
      )}

    </Container>
  );
}

export default NavBar