import React from "react";
import Container from "./Container";
import Button from "./Button";
import Image from "next/image";
import Link from "next/link";

const Hero = () => {
  return (
    <Container>
      <div className="w-full max-w-[1161px] mx-auto py-16 relative">
        <div>
          <p className="font-[Suranna,serif] text-[40px] md:text-[60px] lg:text-[80px] font-normal leading-tight text-black">
            Understand your brain <br />
            - your{" "}
            <span className="text-[#00E58C] italic relative">
              password
              <span className="absolute left-0 bottom-[-12px] w-full h-[6px] bg-[#00E58C] -skew-x-12 opacity-80"></span>
            </span>{" "}
            to the world.
          </p>
          <p className="mt-6 text-gray-600 text-lg max-w-lg">
            We use your brainwaves to recognize who you are — no passwords,
            just you.
          </p>
          <div className="flex gap-4 mt-10">
            <Link href={'/signin'}>
              <Button>Sign In</Button>
            </Link>
            <Link href={'/signup'} >
              <Button outline>Sign Up</Button>
            </Link>
          </div>
        </div>

        <div className="absolute -z-99 top-[10%] right-0 w-3/4 h-full">
          {/* Background world map */}
          <Image
            src="/map.png" // 👉 thay bằng hình map thật
            alt="World Map"
            fill
            className="object-contain"
          />
          {/* Bạn có thể thêm line/curve SVG để connect các avatar giống Figma */}
        </div>
      </div>
    </Container>
  );
};

export default Hero;
