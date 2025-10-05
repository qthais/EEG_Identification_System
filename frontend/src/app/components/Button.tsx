import React from 'react'
interface ButtonProps {
    children: React.ReactNode,
    width?: number,
    outline?: boolean,
    onClick?: () => void,
    className?: string,
    type?: "button" | "submit" | "reset";
    disabled?: boolean;
}
const Button: React.FC<ButtonProps> = ({
    children,
    width,
    outline = false,
    onClick,
    className,
    type = "button",
    disabled = false,
}) => {
    const baseStyle    = "px-6 py-3 rounded-full text-sm font-medium transition cursor-pointer";
    const outlineStyle = "border border-black text-black hover:bg-black hover:text-white";
    const filledStyle  = "bg-black text-white hover:bg-pink-700";
    return (
        <button
            type={type}
            onClick={onClick}
            style={width ? {width: `${width}px`}:{}}
            className={`${baseStyle} ${outline ? outlineStyle : filledStyle} ${className}`}
            disabled={disabled}
        >
            {children}
        </button>
    )
}

export default Button