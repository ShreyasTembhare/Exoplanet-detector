import "@astryxdesign/core/reset.css";
import "@astryxdesign/core/astryx.css";
import { createRoot } from "react-dom/client";
import { Theme, defineTheme } from "@astryxdesign/core/theme";
import { themes, defaultThemeSlug } from "./astryx-themes";
import App from "./App.jsx";

const importedTheme = themes[defaultThemeSlug];

const deskTheme = defineTheme({
  name: "transit-desk",
  extends: importedTheme,
  tokens: {
    "--radius-element": "10px",
    "--radius-container": "14px",
  },
  motion: { character: "crisp" },
});

createRoot(document.getElementById("root")).render(
  <Theme theme={deskTheme} mode="light">
    <App />
  </Theme>,
);
