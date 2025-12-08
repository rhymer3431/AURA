import svgPaths from "./svg-1571uplucg";
import imgCreditsToUnsplashCom1 from "figma:asset/e07e0180c8f688e34c197dd4674fd7e62937b68c.png";
import imgCreditsToUnsplashCom2 from "figma:asset/93539f7e33b5ee1ed96651e2cef8e4e1bc7c5019.png";
import imgCreditsToUnsplashCom3 from "figma:asset/81317c1a6461999d545aa891a9277a99295c83d4.png";
import imgCreditsToUnsplashCom4 from "figma:asset/59a714b86b104a52903f0b67ed801ccac9993666.png";
import imgCreditsToUnsplashCom5 from "figma:asset/58e2826c9ef36e80f373f0ff9cb6ba2ea2a3d704.png";
import imgCreditsToUnsplashCom6 from "figma:asset/b940caf9f3a52bcc9317c793ebc094db911b237b.png";
import { imgCreditsToUnsplashCom, imgGroup1 } from "./svg-yxh8w";

function Background() {
  return (
    <div className="absolute contents left-px top-0" data-name="Background">
      <div className="absolute bg-[#f8f9fa] h-[1137px] left-px top-0 w-[1919px]" />
    </div>
  );
}

function Menu() {
  return (
    <div className="absolute contents font-['Helvetica:Regular',sans-serif] leading-[1.5] left-[1575.5px] not-italic text-[#a0aec0] text-[12px] top-[1095.5px]" data-name="Menu">
      <p className="absolute h-[18px] left-[1856.5px] top-[1095.5px] w-[41.5px]">License</p>
      <p className="absolute h-[18px] left-[1788px] top-[1095.5px] w-[24.5px]">Blog</p>
      <p className="absolute h-[18px] left-[1687px] top-[1095.5px] w-[57px]">Simmmple</p>
      <p className="absolute h-[18px] left-[1575.5px] top-[1095.5px] w-[67.5px]">Creative Tim</p>
    </div>
  );
}

function Copyright() {
  return (
    <div className="absolute contents left-[298px] top-[1095.5px]" data-name="Copyright">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[18px] leading-[1.5] left-[298px] not-italic text-[#a0aec0] text-[0px] text-[12px] top-[1095.5px] w-[372.5px]">
        <span>{`@ 2021, Made with ❤️ by `}</span>
        <span className="font-['Helvetica:Bold',sans-serif] text-[#38b2ac]">Creative Tim</span>
        <span>{` & `}</span>
        <span className="font-['Helvetica:Bold',sans-serif] text-[#38b2ac]">Simmmple</span>
        <span>{` for a better web`}</span>
      </p>
    </div>
  );
}

function FooterMenu() {
  return (
    <div className="absolute contents left-[298px] top-[1095.5px]" data-name="Footer Menu">
      <Menu />
      <Copyright />
    </div>
  );
}

function Background1() {
  return (
    <div className="absolute contents left-[298px] top-[613.5px]" data-name="Background">
      <div className="absolute bg-white h-[453.5px] left-[298px] rounded-[15px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] top-[613.5px] w-[1600px]" />
    </div>
  );
}

function MoreVert() {
  return (
    <div className="absolute left-[1782.5px] size-[20px] top-[1017px]" data-name="more_vert">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 20 20">
        <g clipPath="url(#clip0_1_642)" id="more_vert">
          <g id="Vector"></g>
          <path d={svgPaths.p24b71d80} fill="var(--fill-0, #A0AEC0)" id="Vector_2" />
        </g>
        <defs>
          <clipPath id="clip0_1_642">
            <rect fill="white" height="20" width="20" />
          </clipPath>
        </defs>
      </svg>
    </div>
  );
}

function Icon() {
  return (
    <div className="absolute contents left-[1782.5px] top-[1017px]" data-name="Icon">
      <MoreVert />
    </div>
  );
}

function Progress() {
  return (
    <div className="absolute contents left-[1551.5px] top-[1015px]" data-name="Progress">
      <div className="absolute h-0 left-[1551.5px] top-[1039px] w-[125px]">
        <div className="absolute bottom-0 left-0 right-0 top-[-3px]">
          <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 125 3">
            <line id="Line 2" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" x1="1.5" x2="123.5" y1="1.5" y2="1.5" />
          </svg>
        </div>
      </div>
      <div className="absolute h-0 left-[1551.5px] top-[1039px] w-[36px]">
        <div className="absolute bottom-0 left-0 right-0 top-[-3px]">
          <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 36 3">
            <line id="Line 3" stroke="var(--stroke-0, #4FD1C5)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" x1="1.5" x2="34.5" y1="1.5" y2="1.5" />
          </svg>
        </div>
      </div>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1551.5px] not-italic text-[#4fd1c5] text-[14px] top-[1015px] w-[28.5px]">25%</p>
    </div>
  );
}

function Status() {
  return (
    <div className="absolute contents left-[1261.5px] top-[1017.5px]" data-name="Status">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1289.5px] not-italic text-[#2d3748] text-[14px] text-center top-[1017.5px] translate-x-[-50%] w-[56px]">Working</p>
    </div>
  );
}

function Budget() {
  return (
    <div className="absolute contents left-[984.5px] top-[1017.5px]" data-name="Budget">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[984.5px] not-italic text-[#2d3748] text-[14px] top-[1017.5px] w-[31.5px]">$400</p>
    </div>
  );
}

function Title() {
  return (
    <div className="absolute contents left-[357.5px] top-[1017.5px]" data-name="Title">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[357.5px] not-italic text-[#2d3748] text-[14px] top-[1017.5px] w-[174px]">Add the New Pricing Page</p>
    </div>
  );
}

function Jira() {
  return (
    <div className="absolute inset-[89.4%_82.27%_8.75%_16.69%]" data-name="jira-3 1">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 20 21">
        <g clipPath="url(#clip0_1_614)" id="jira-3 1">
          <path d={svgPaths.p33dae880} fill="var(--fill-0, #2684FF)" id="Vector" />
          <path d={svgPaths.p11af1280} fill="url(#paint0_linear_1_614)" id="Vector_2" />
          <path d={svgPaths.p2664c300} fill="url(#paint1_linear_1_614)" id="Vector_3" />
        </g>
        <defs>
          <linearGradient gradientUnits="userSpaceOnUse" id="paint0_linear_1_614" x1="9.49208" x2="5.20782" y1="4.32768" y2="8.57919">
            <stop offset="0.18" stopColor="#0052CC" />
            <stop offset="1" stopColor="#2684FF" />
          </linearGradient>
          <linearGradient gradientUnits="userSpaceOnUse" id="paint1_linear_1_614" x1="776.103" x2="1212.24" y1="900.925" y2="1181.57">
            <stop offset="0.18" stopColor="#0052CC" />
            <stop offset="1" stopColor="#2684FF" />
          </linearGradient>
          <clipPath id="clip0_1_614">
            <rect fill="white" height="21" width="20" />
          </clipPath>
        </defs>
      </svg>
    </div>
  );
}

function Icon1() {
  return (
    <div className="absolute contents inset-[89.4%_82.27%_8.75%_16.69%]" data-name="Icon">
      <Jira />
    </div>
  );
}

function Content() {
  return (
    <div className="absolute contents left-[320.5px] top-[1015px]" data-name="Content">
      <Icon />
      <Progress />
      <Status />
      <Budget />
      <Title />
      <Icon1 />
    </div>
  );
}

function NewPricingPage() {
  return (
    <div className="absolute contents left-[320.5px] top-[1015px]" data-name="New Pricing Page">
      <Content />
    </div>
  );
}

function Line() {
  return (
    <div className="absolute h-0 left-[320.5px] top-[996px] w-[1555px]" data-name="Line">
      <div className="absolute inset-[-0.5px_-0.03%]">
        <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 1556 1">
          <g id="Line">
            <path d="M0.5 0.5H1555.5" id="Vector 5" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </svg>
      </div>
    </div>
  );
}

function MoreVert1() {
  return (
    <div className="absolute left-[1782.5px] size-[20px] top-[955px]" data-name="more_vert">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 20 20">
        <g clipPath="url(#clip0_1_642)" id="more_vert">
          <g id="Vector"></g>
          <path d={svgPaths.p24b71d80} fill="var(--fill-0, #A0AEC0)" id="Vector_2" />
        </g>
        <defs>
          <clipPath id="clip0_1_642">
            <rect fill="white" height="20" width="20" />
          </clipPath>
        </defs>
      </svg>
    </div>
  );
}

function Icon2() {
  return (
    <div className="absolute contents left-[1782.5px] top-[955px]" data-name="Icon">
      <MoreVert1 />
    </div>
  );
}

function Progress1() {
  return (
    <div className="absolute contents left-[1551.5px] top-[953px]" data-name="Progress">
      <div className="absolute h-0 left-[1551.5px] top-[977px] w-[125px]">
        <div className="absolute bottom-0 left-0 right-0 top-[-3px]">
          <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 125 3">
            <line id="Line 2" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" x1="1.5" x2="123.5" y1="1.5" y2="1.5" />
          </svg>
        </div>
      </div>
      <div className="absolute h-0 left-[1551.5px] top-[977px] w-[125px]">
        <div className="absolute bottom-0 left-0 right-0 top-[-3px]">
          <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 125 3">
            <line id="Line 3" stroke="var(--stroke-0, #4FD1C5)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" x1="1.5" x2="123.5" y1="1.5" y2="1.5" />
          </svg>
        </div>
      </div>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1551.5px] not-italic text-[#4fd1c5] text-[14px] top-[953px] w-[36px]">100%</p>
    </div>
  );
}

function Status1() {
  return (
    <div className="absolute contents left-[1272px] top-[955.5px]" data-name="Status">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1289.75px] not-italic text-[#2d3748] text-[14px] text-center top-[955.5px] translate-x-[-50%] w-[35.5px]">Done</p>
    </div>
  );
}

function Budget1() {
  return (
    <div className="absolute contents left-[984.5px] top-[955.5px]" data-name="Budget">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[984.5px] not-italic text-[#2d3748] text-[14px] top-[955.5px] w-[51px]">$32,000</p>
    </div>
  );
}

function Title1() {
  return (
    <div className="absolute contents left-[357.5px] top-[955.5px]" data-name="Title">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[357.5px] not-italic text-[#2d3748] text-[14px] top-[955.5px] w-[155.5px]">Launch our Mobile App</p>
    </div>
  );
}

function Spotify() {
  return (
    <div className="absolute inset-[83.99%_82.27%_14.25%_16.69%]" data-name="spotify-2 1">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 20 20">
        <g clipPath="url(#clip0_1_582)" id="spotify-2 1">
          <path d={svgPaths.p3b151200} fill="var(--fill-0, #2EBD59)" id="Vector" />
        </g>
        <defs>
          <clipPath id="clip0_1_582">
            <rect fill="white" height="20" width="20" />
          </clipPath>
        </defs>
      </svg>
    </div>
  );
}

function Icon3() {
  return (
    <div className="absolute contents inset-[83.99%_82.27%_14.25%_16.69%]" data-name="Icon">
      <Spotify />
    </div>
  );
}

function Content1() {
  return (
    <div className="absolute contents left-[320.5px] top-[953px]" data-name="Content">
      <Icon2 />
      <Progress1 />
      <Status1 />
      <Budget1 />
      <Title1 />
      <Icon3 />
    </div>
  );
}

function LaunchOurMobileApp() {
  return (
    <div className="absolute contents left-[320.5px] top-[953px]" data-name="Launch our Mobile App">
      <Line />
      <Content1 />
    </div>
  );
}

function Line1() {
  return (
    <div className="absolute h-0 left-[320.5px] top-[934px] w-[1555px]" data-name="Line">
      <div className="absolute inset-[-0.5px_-0.03%]">
        <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 1556 1">
          <g id="Line">
            <path d="M0.5 0.5H1555.5" id="Vector 5" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </svg>
      </div>
    </div>
  );
}

function MoreVert2() {
  return (
    <div className="absolute left-[1782.5px] size-[20px] top-[893px]" data-name="more_vert">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 20 20">
        <g clipPath="url(#clip0_1_642)" id="more_vert">
          <g id="Vector"></g>
          <path d={svgPaths.p24b71d80} fill="var(--fill-0, #A0AEC0)" id="Vector_2" />
        </g>
        <defs>
          <clipPath id="clip0_1_642">
            <rect fill="white" height="20" width="20" />
          </clipPath>
        </defs>
      </svg>
    </div>
  );
}

function Icon4() {
  return (
    <div className="absolute contents left-[1782.5px] top-[893px]" data-name="Icon">
      <MoreVert2 />
    </div>
  );
}

function Progress2() {
  return (
    <div className="absolute contents left-[1551.5px] top-[891px]" data-name="Progress">
      <div className="absolute h-0 left-[1551.5px] top-[915px] w-[125px]">
        <div className="absolute bottom-0 left-0 right-0 top-[-3px]">
          <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 125 3">
            <line id="Line 2" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" x1="1.5" x2="123.5" y1="1.5" y2="1.5" />
          </svg>
        </div>
      </div>
      <div className="absolute h-0 left-[1551.5px] top-[915px] w-[125px]">
        <div className="absolute bottom-0 left-0 right-0 top-[-3px]">
          <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 125 3">
            <line id="Line 3" stroke="var(--stroke-0, #4FD1C5)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" x1="1.5" x2="123.5" y1="1.5" y2="1.5" />
          </svg>
        </div>
      </div>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1551.5px] not-italic text-[#4fd1c5] text-[14px] top-[891px] w-[36px]">100%</p>
    </div>
  );
}

function Status2() {
  return (
    <div className="absolute contents left-[1272px] top-[893.5px]" data-name="Status">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1289.75px] not-italic text-[#2d3748] text-[14px] text-center top-[893.5px] translate-x-[-50%] w-[35.5px]">Done</p>
    </div>
  );
}

function Budget2() {
  return (
    <div className="absolute contents left-[984.5px] top-[893.5px]" data-name="Budget">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[984.5px] not-italic text-[#2d3748] text-[14px] top-[893.5px] w-[47.5px]">Not set</p>
    </div>
  );
}

function Title2() {
  return (
    <div className="absolute contents left-[357.5px] top-[893.5px]" data-name="Title">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[357.5px] not-italic text-[#2d3748] text-[14px] top-[893.5px] w-[127px]">Fix Platform Errors</p>
    </div>
  );
}

function SlackNewLogo() {
  return (
    <div className="absolute h-[20.5px] left-[320.5px] overflow-clip top-[892.5px] w-[20px]" data-name="slack-new-logo 1">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 20 21">
        <g id="Group">
          <path clipRule="evenodd" d={svgPaths.p30ec6400} fill="var(--fill-0, #36C5F0)" fillRule="evenodd" id="Vector" />
          <path clipRule="evenodd" d={svgPaths.p1d1bfa00} fill="var(--fill-0, #2EB67D)" fillRule="evenodd" id="Vector_2" />
          <path clipRule="evenodd" d={svgPaths.p13149f00} fill="var(--fill-0, #ECB22E)" fillRule="evenodd" id="Vector_3" />
          <path clipRule="evenodd" d={svgPaths.p32889400} fill="var(--fill-0, #E01E5A)" fillRule="evenodd" id="Vector_4" />
        </g>
      </svg>
    </div>
  );
}

function Icon5() {
  return (
    <div className="absolute contents left-[320.5px] top-[892.5px]" data-name="Icon">
      <SlackNewLogo />
    </div>
  );
}

function Content2() {
  return (
    <div className="absolute contents left-[320.5px] top-[891px]" data-name="Content">
      <Icon4 />
      <Progress2 />
      <Status2 />
      <Budget2 />
      <Title2 />
      <Icon5 />
    </div>
  );
}

function FixPlatformErrors() {
  return (
    <div className="absolute contents left-[320.5px] top-[891px]" data-name="Fix Platform Errors">
      <Line1 />
      <Content2 />
    </div>
  );
}

function Line2() {
  return (
    <div className="absolute h-0 left-[320.5px] top-[872px] w-[1555px]" data-name="Line">
      <div className="absolute inset-[-0.5px_-0.03%]">
        <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 1556 1">
          <g id="Line">
            <path d="M0.5 0.5H1555.5" id="Vector 5" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </svg>
      </div>
    </div>
  );
}

function MoreVert3() {
  return (
    <div className="absolute left-[1782.5px] size-[20px] top-[831px]" data-name="more_vert">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 20 20">
        <g clipPath="url(#clip0_1_642)" id="more_vert">
          <g id="Vector"></g>
          <path d={svgPaths.p24b71d80} fill="var(--fill-0, #A0AEC0)" id="Vector_2" />
        </g>
        <defs>
          <clipPath id="clip0_1_642">
            <rect fill="white" height="20" width="20" />
          </clipPath>
        </defs>
      </svg>
    </div>
  );
}

function Icon6() {
  return (
    <div className="absolute contents left-[1782.5px] top-[831px]" data-name="Icon">
      <MoreVert3 />
    </div>
  );
}

function Progress3() {
  return (
    <div className="absolute contents left-[1551.5px] top-[829px]" data-name="Progress">
      <div className="absolute h-0 left-[1551.5px] top-[853px] w-[125px]">
        <div className="absolute bottom-0 left-0 right-0 top-[-3px]">
          <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 125 3">
            <line id="Line 2" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" x1="1.5" x2="123.5" y1="1.5" y2="1.5" />
          </svg>
        </div>
      </div>
      <div className="absolute h-0 left-[1551.5px] top-[853px] w-[18.5px]">
        <div className="absolute bottom-0 left-0 right-0 top-[-3px]">
          <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 19 3">
            <line id="Line 3" stroke="var(--stroke-0, #4FD1C5)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" x1="1.5" x2="17" y1="1.5" y2="1.5" />
          </svg>
        </div>
      </div>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1551.5px] not-italic text-[#4fd1c5] text-[14px] top-[829px] w-[28.5px]">10%</p>
    </div>
  );
}

function Status3() {
  return (
    <div className="absolute contents left-[1258.5px] top-[831.5px]" data-name="Status">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1289.75px] not-italic text-[#2d3748] text-[14px] text-center top-[831.5px] translate-x-[-50%] w-[62.5px]">Canceled</p>
    </div>
  );
}

function Budget3() {
  return (
    <div className="absolute contents left-[984.5px] top-[831.5px]" data-name="Budget">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[984.5px] not-italic text-[#2d3748] text-[14px] top-[831.5px] w-[43px]">$3,000</p>
    </div>
  );
}

function Title3() {
  return (
    <div className="absolute contents left-[357.5px] top-[831.5px]" data-name="Title">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[357.5px] not-italic text-[#2d3748] text-[14px] top-[831.5px] w-[132.5px]">Add Progress Track</p>
    </div>
  );
}

function Group2() {
  return (
    <div className="absolute inset-[73.04%_82.27%_25.2%_16.69%]">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 20 20">
        <g id="Group 3">
          <path d={svgPaths.pc0aee00} fill="url(#paint0_linear_1_591)" id="Vector" />
          <path d={svgPaths.pa08c200} fill="var(--fill-0, #2684FF)" id="Vector_2" />
        </g>
        <defs>
          <linearGradient gradientUnits="userSpaceOnUse" id="paint0_linear_1_591" x1="8.62424" x2="3.44981" y1="10.7373" y2="19.6995">
            <stop stopColor="#0052CC" />
            <stop offset="0.92" stopColor="#2684FF" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
}

function Icon7() {
  return (
    <div className="absolute contents inset-[73.04%_82.27%_25.2%_16.69%]" data-name="Icon">
      <Group2 />
    </div>
  );
}

function Content3() {
  return (
    <div className="absolute contents left-[320.5px] top-[829px]" data-name="Content">
      <Icon6 />
      <Progress3 />
      <Status3 />
      <Budget3 />
      <Title3 />
      <Icon7 />
    </div>
  );
}

function ProgressTrack() {
  return (
    <div className="absolute contents left-[320.5px] top-[829px]" data-name="Progress Track">
      <Line2 />
      <Content3 />
    </div>
  );
}

function Line3() {
  return (
    <div className="absolute h-0 left-[320.5px] top-[810px] w-[1555px]" data-name="Line">
      <div className="absolute inset-[-0.5px_-0.03%]">
        <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 1556 1">
          <g id="Line">
            <path d="M0.5 0.5H1555.5" id="Vector 5" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </svg>
      </div>
    </div>
  );
}

function MoreVert4() {
  return (
    <div className="absolute left-[1782.5px] size-[20px] top-[769px]" data-name="more_vert">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 20 20">
        <g clipPath="url(#clip0_1_642)" id="more_vert">
          <g id="Vector"></g>
          <path d={svgPaths.p24b71d80} fill="var(--fill-0, #A0AEC0)" id="Vector_2" />
        </g>
        <defs>
          <clipPath id="clip0_1_642">
            <rect fill="white" height="20" width="20" />
          </clipPath>
        </defs>
      </svg>
    </div>
  );
}

function Icon8() {
  return (
    <div className="absolute contents left-[1782.5px] top-[769px]" data-name="Icon">
      <MoreVert4 />
    </div>
  );
}

function Progress4() {
  return (
    <div className="absolute contents left-[1552px] top-[767px]" data-name="Progress">
      <div className="absolute h-0 left-[1552px] top-[791px] w-[125px]">
        <div className="absolute bottom-0 left-0 right-0 top-[-3px]">
          <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 125 3">
            <line id="Line 2" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" x1="1.5" x2="123.5" y1="1.5" y2="1.5" />
          </svg>
        </div>
      </div>
      <div className="absolute h-0 left-[1552px] top-[791px] w-[82.5px]">
        <div className="absolute bottom-0 left-0 right-0 top-[-3px]">
          <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 83 3">
            <line id="Line 3" stroke="var(--stroke-0, #4FD1C5)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" x1="1.5" x2="81" y1="1.5" y2="1.5" />
          </svg>
        </div>
      </div>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1552px] not-italic text-[#4fd1c5] text-[14px] top-[767px] w-[28.5px]">60%</p>
    </div>
  );
}

function Status4() {
  return (
    <div className="absolute contents left-[1261.5px] top-[769.5px]" data-name="Status">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1289.5px] not-italic text-[#2d3748] text-[14px] text-center top-[769.5px] translate-x-[-50%] w-[56px]">Working</p>
    </div>
  );
}

function Budget4() {
  return (
    <div className="absolute contents left-[984.5px] top-[769.5px]" data-name="Budget">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[984.5px] not-italic text-[#2d3748] text-[14px] top-[769.5px] w-[51px]">$14,000</p>
    </div>
  );
}

function Title4() {
  return (
    <div className="absolute contents left-[357.5px] top-[769.5px]" data-name="Title">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[357.5px] not-italic text-[#2d3748] text-[14px] top-[769.5px] w-[151.5px]">Chakra Soft UI Version</p>
    </div>
  );
}

function UiUxSurface() {
  return (
    <div className="absolute contents inset-0" data-name="UI UX Surface">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 20 20">
        <g id="Outline no shadow">
          <path d={svgPaths.p29263180} fill="var(--fill-0, #470137)" id="Vector" />
        </g>
      </svg>
    </div>
  );
}

function Surfaces() {
  return (
    <div className="absolute contents inset-0" data-name="Surfaces">
      <UiUxSurface />
    </div>
  );
}

function Xd() {
  return (
    <div className="absolute inset-[21.92%_14.54%_28.63%_14.57%]" data-name="Xd">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 15 10">
        <g id="Xd">
          <path d={svgPaths.p163b1c80} fill="var(--fill-0, #FF61F6)" id="Vector" />
          <path d={svgPaths.p3982b3c0} fill="var(--fill-0, #FF61F6)" id="Vector_2" />
        </g>
      </svg>
    </div>
  );
}

function OutlinedMnemonicsLogos() {
  return (
    <div className="absolute contents inset-[21.92%_14.54%_28.63%_14.57%]" data-name="Outlined Mnemonics Logos">
      <Xd />
    </div>
  );
}

function Layer() {
  return (
    <div className="absolute contents inset-0" data-name="Layer 2 1">
      <Surfaces />
      <OutlinedMnemonicsLogos />
    </div>
  );
}

function AdobeXdCcIcon() {
  return (
    <div className="absolute h-[19.5px] left-[320.5px] overflow-clip top-[769px] w-[20px]" data-name="Adobe_XD_CC_icon 1">
      <Layer />
    </div>
  );
}

function Icon9() {
  return (
    <div className="absolute contents left-[320.5px] top-[769px]" data-name="Icon">
      <AdobeXdCcIcon />
    </div>
  );
}

function Content4() {
  return (
    <div className="absolute contents left-[320.5px] top-[767px]" data-name="Content">
      <Icon8 />
      <Progress4 />
      <Status4 />
      <Budget4 />
      <Title4 />
      <Icon9 />
    </div>
  );
}

function SoftUiXd() {
  return (
    <div className="absolute contents left-[320.5px] top-[767px]" data-name="Soft UI XD">
      <Line3 />
      <Content4 />
    </div>
  );
}

function Items() {
  return (
    <div className="absolute contents left-[320.5px] top-[767px]" data-name="Items">
      <NewPricingPage />
      <LaunchOurMobileApp />
      <FixPlatformErrors />
      <ProgressTrack />
      <SoftUiXd />
    </div>
  );
}

function Lines() {
  return (
    <div className="absolute h-0 left-[320.5px] top-[748px] w-[1555px]" data-name="Lines">
      <div className="absolute inset-[-0.5px_-0.03%]">
        <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 1556 1">
          <g id="Line">
            <path d="M0.5 0.5H1555.5" id="Vector 5" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </svg>
      </div>
    </div>
  );
}

function Titles() {
  return (
    <div className="absolute contents font-['Helvetica:Bold',sans-serif] leading-[1.5] left-[320.5px] not-italic text-[#a0aec0] text-[10px] top-[720.5px]" data-name="Titles">
      <p className="absolute h-[15px] left-[1552px] top-[720.5px] w-[67px]">COMPLETION</p>
      <p className="absolute h-[15px] left-[1270px] top-[720.5px] w-[39px]">STATUS</p>
      <p className="absolute h-[15px] left-[984.5px] top-[720.5px] w-[42.5px]">BUDGET</p>
      <p className="absolute h-[15px] left-[320.5px] top-[720.5px] w-[60px]">COMPANIES</p>
    </div>
  );
}

function List() {
  return (
    <div className="absolute contents left-[320.5px] top-[720.5px]" data-name="List">
      <Items />
      <Lines />
      <Titles />
    </div>
  );
}

function IonIconCCheckmarkCCircle() {
  return (
    <div className="absolute left-[319px] size-[15px] top-[674.5px]" data-name="IONIcon/C/checkmark/C/circle">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 15 15">
        <g id="IONIcon/C/checkmark/C/circle">
          <path d={svgPaths.p22b03640} fill="var(--fill-0, #68D391)" id="Vector" />
        </g>
      </svg>
    </div>
  );
}

function Text() {
  return (
    <div className="absolute contents left-[319px] top-[641.5px]" data-name="Text">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] leading-[1.4] left-[338.5px] not-italic text-[#a0aec0] text-[0px] text-[14px] top-[672.5px] w-[121.5px]">
        <span className="font-['Helvetica:Bold',sans-serif]">30 done</span>
        <span>{` this month`}</span>
      </p>
      <IonIconCCheckmarkCCircle />
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[25px] leading-[1.4] left-[319px] not-italic text-[#2d3748] text-[18px] top-[641.5px] w-[71.5px]">Projects</p>
    </div>
  );
}

function SecondCard() {
  return (
    <div className="absolute contents left-[298px] top-[613.5px]" data-name="Second Card">
      <Background1 />
      <List />
      <Text />
    </div>
  );
}

function Background2() {
  return (
    <div className="absolute contents left-[298px] top-[101px]" data-name="Background">
      <div className="absolute bg-white h-[488.5px] left-[298px] rounded-[15px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] top-[101px] w-[1600px]" />
    </div>
  );
}

function Edit() {
  return (
    <div className="absolute contents left-[1781px] top-[537px]" data-name="Edit">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[18px] leading-[1.5] left-[1781px] not-italic text-[#718096] text-[12px] top-[537px] w-[23px]">Edit</p>
    </div>
  );
}

function Employed() {
  return (
    <div className="absolute contents left-[1552px] top-[536.5px]" data-name="Employed">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1579.5px] not-italic text-[#2d3748] text-[14px] text-center top-[536.5px] translate-x-[-50%] w-[55px]">14/06/21</p>
    </div>
  );
}

function Status5() {
  return (
    <div className="absolute contents left-[1305.5px] top-[533.5px]" data-name="Status">
      <div className="absolute bg-[#cbd5e0] h-[25px] left-[1305.5px] rounded-[8px] top-[533.5px] w-[65px]" />
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1338.25px] not-italic text-[14px] text-center text-white top-[536px] translate-x-[-50%] w-[44.5px]">Offline</p>
    </div>
  );
}

function Function() {
  return (
    <div className="absolute contents leading-[1.4] left-[1045px] not-italic text-[14px] top-[528px]" data-name="Function">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] left-[1045px] text-[#718096] top-[545px] w-[85px]">UI/UX Design</p>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] left-[1045px] text-[#2d3748] top-[528px] w-[60px]">Designer</p>
    </div>
  );
}

function Name() {
  return (
    <div className="absolute contents leading-[1.4] left-[375.5px] not-italic text-[14px] top-[528px]" data-name="Name">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] left-[375.5px] text-[#718096] top-[545px] w-[139.5px]">mark@simmmple.com</p>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] left-[375.5px] text-[#2d3748] top-[528px] w-[82.5px]">Mark Wilson</p>
    </div>
  );
}

function Image() {
  return (
    <div className="absolute contents left-[320.5px] top-[526px]" data-name="Image">
      <div className="absolute bg-[#4fd1c5] left-[320.5px] rounded-[12px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] size-[40px] top-[526px]" />
      <div className="absolute inset-[45.91%_81.17%_49.03%_16.46%] mask-alpha mask-intersect mask-no-clip mask-no-repeat mask-position-[-1px_2px] mask-size-[51px_51px]" data-name="Credits to Unsplash.com" style={{ maskImage: `url('${imgCreditsToUnsplashCom}')` }}>
        <img alt="" className="absolute inset-0 max-w-none object-50%-50% object-cover pointer-events-none size-full" src={imgCreditsToUnsplashCom1} />
      </div>
    </div>
  );
}

function Avatar() {
  return (
    <div className="absolute contents left-[320.5px] top-[526px]" data-name="Avatar">
      <Name />
      <Image />
    </div>
  );
}

function Content5() {
  return (
    <div className="absolute contents left-[320.5px] top-[526px]" data-name="Content">
      <Edit />
      <Employed />
      <Status5 />
      <Function />
      <Avatar />
    </div>
  );
}

function MarkWilson() {
  return (
    <div className="absolute contents left-[320.5px] top-[526px]" data-name="Mark Wilson">
      <Content5 />
    </div>
  );
}

function Line4() {
  return (
    <div className="absolute h-0 left-[320.5px] top-[515px] w-[1555px]" data-name="Line">
      <div className="absolute inset-[-0.5px_-0.03%]">
        <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 1556 1">
          <g id="Line">
            <path d="M0.5 0.5H1555.5" id="Vector 5" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </svg>
      </div>
    </div>
  );
}

function Edit1() {
  return (
    <div className="absolute contents left-[1781px] top-[475px]" data-name="Edit">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[18px] leading-[1.5] left-[1781px] not-italic text-[#718096] text-[12px] top-[475px] w-[23px]">Edit</p>
    </div>
  );
}

function Employed1() {
  return (
    <div className="absolute contents left-[1552px] top-[474.5px]" data-name="Employed">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1579.5px] not-italic text-[#2d3748] text-[14px] text-center top-[474.5px] translate-x-[-50%] w-[55px]">14/06/21</p>
    </div>
  );
}

function Status6() {
  return (
    <div className="absolute contents left-[1305.5px] top-[471.5px]" data-name="Status">
      <div className="absolute bg-[#cbd5e0] h-[25px] left-[1305.5px] rounded-[8px] top-[471.5px] w-[65px]" />
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1338.25px] not-italic text-[14px] text-center text-white top-[474px] translate-x-[-50%] w-[44.5px]">Offline</p>
    </div>
  );
}

function Function1() {
  return (
    <div className="absolute contents leading-[1.4] left-[1045px] not-italic text-[14px] top-[466px]" data-name="Function">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] left-[1045px] text-[#718096] top-[483px] w-[64px]">Developer</p>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] left-[1045px] text-[#2d3748] top-[466px] w-[83.5px]">Programmer</p>
    </div>
  );
}

function Name1() {
  return (
    <div className="absolute contents leading-[1.4] left-[375.5px] not-italic text-[14px] top-[466px]" data-name="Name">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] left-[375.5px] text-[#718096] top-[483px] w-[146px]">daniel@simmmple.com</p>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] left-[375.5px] text-[#2d3748] top-[466px] w-[100px]">Daniel Thomas</p>
    </div>
  );
}

function Image1() {
  return (
    <div className="absolute contents left-[320.5px] top-[464px]" data-name="Image">
      <div className="absolute bg-[#4fd1c5] left-[320.5px] rounded-[12px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] size-[40px] top-[464px]" />
      <div className="absolute inset-[40.46%_81.17%_54.49%_16.46%] mask-alpha mask-intersect mask-no-clip mask-no-repeat mask-position-[-1px_2px] mask-size-[51px_51px]" data-name="Credits to Unsplash.com" style={{ maskImage: `url('${imgCreditsToUnsplashCom}')` }}>
        <img alt="" className="absolute inset-0 max-w-none object-50%-50% object-cover pointer-events-none size-full" src={imgCreditsToUnsplashCom2} />
      </div>
    </div>
  );
}

function Avatar1() {
  return (
    <div className="absolute contents left-[320.5px] top-[464px]" data-name="Avatar">
      <Name1 />
      <Image1 />
    </div>
  );
}

function Content6() {
  return (
    <div className="absolute contents left-[320.5px] top-[464px]" data-name="Content">
      <Edit1 />
      <Employed1 />
      <Status6 />
      <Function1 />
      <Avatar1 />
    </div>
  );
}

function DanielThomas() {
  return (
    <div className="absolute contents left-[320.5px] top-[464px]" data-name="Daniel Thomas">
      <Line4 />
      <Content6 />
    </div>
  );
}

function Line5() {
  return (
    <div className="absolute h-0 left-[320.5px] top-[453px] w-[1555px]" data-name="Line">
      <div className="absolute inset-[-0.5px_-0.03%]">
        <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 1556 1">
          <g id="Line">
            <path d="M0.5 0.5H1555.5" id="Vector 5" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </svg>
      </div>
    </div>
  );
}

function Edit2() {
  return (
    <div className="absolute contents left-[1781px] top-[413px]" data-name="Edit">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[18px] leading-[1.5] left-[1781px] not-italic text-[#718096] text-[12px] top-[413px] w-[23px]">Edit</p>
    </div>
  );
}

function Employed2() {
  return (
    <div className="absolute contents left-[1552px] top-[412.5px]" data-name="Employed">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1579.5px] not-italic text-[#2d3748] text-[14px] text-center top-[412.5px] translate-x-[-50%] w-[55px]">14/06/21</p>
    </div>
  );
}

function Status7() {
  return (
    <div className="absolute contents left-[1305.5px] top-[409.5px]" data-name="Status">
      <div className="absolute bg-[#48bb78] h-[25px] left-[1305.5px] rounded-[8px] top-[409.5px] w-[65px]" />
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1338px] not-italic text-[14px] text-center text-white top-[412px] translate-x-[-50%] w-[44px]">Online</p>
    </div>
  );
}

function Function2() {
  return (
    <div className="absolute contents leading-[1.4] left-[1045px] not-italic text-[14px] top-[404px]" data-name="Function">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] left-[1045px] text-[#718096] top-[421px] w-[79.5px]">Organization</p>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] left-[1045px] text-[#2d3748] top-[404px] w-[58px]">Manager</p>
    </div>
  );
}

function Name2() {
  return (
    <div className="absolute contents leading-[1.4] left-[375.5px] not-italic text-[14px] top-[404px]" data-name="Name">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] left-[375.5px] text-[#718096] top-[421px] w-[168.5px]">freduardo@simmmple.com</p>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] left-[375.5px] text-[#2d3748] top-[404px] w-[95px]">Freduardo Hill</p>
    </div>
  );
}

function Image2() {
  return (
    <div className="absolute contents left-[320.5px] top-[402px]" data-name="Image">
      <div className="absolute bg-[#4fd1c5] left-[320.5px] rounded-[12px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] size-[40px] top-[402px]" />
      <div className="absolute inset-[35%_81.17%_59.94%_16.46%] mask-alpha mask-intersect mask-no-clip mask-no-repeat mask-position-[-1px_2px] mask-size-[51px_51px]" data-name="Credits to Unsplash.com" style={{ maskImage: `url('${imgCreditsToUnsplashCom}')` }}>
        <img alt="" className="absolute inset-0 max-w-none object-50%-50% object-cover pointer-events-none size-full" src={imgCreditsToUnsplashCom3} />
      </div>
    </div>
  );
}

function Avatar2() {
  return (
    <div className="absolute contents left-[320.5px] top-[402px]" data-name="Avatar">
      <Name2 />
      <Image2 />
    </div>
  );
}

function Content7() {
  return (
    <div className="absolute contents left-[320.5px] top-[402px]" data-name="Content">
      <Edit2 />
      <Employed2 />
      <Status7 />
      <Function2 />
      <Avatar2 />
    </div>
  );
}

function FreduardoHill() {
  return (
    <div className="absolute contents left-[320.5px] top-[402px]" data-name="Freduardo Hill">
      <Line5 />
      <Content7 />
    </div>
  );
}

function Line6() {
  return (
    <div className="absolute h-0 left-[320.5px] top-[391px] w-[1555px]" data-name="Line">
      <div className="absolute inset-[-0.5px_-0.03%]">
        <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 1556 1">
          <g id="Line">
            <path d="M0.5 0.5H1555.5" id="Vector 5" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </svg>
      </div>
    </div>
  );
}

function Edit3() {
  return (
    <div className="absolute contents left-[1781px] top-[351px]" data-name="Edit">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[18px] leading-[1.5] left-[1781px] not-italic text-[#718096] text-[12px] top-[351px] w-[23px]">Edit</p>
    </div>
  );
}

function Employed3() {
  return (
    <div className="absolute contents left-[1552px] top-[350.5px]" data-name="Employed">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1579.5px] not-italic text-[#2d3748] text-[14px] text-center top-[350.5px] translate-x-[-50%] w-[55px]">14/06/21</p>
    </div>
  );
}

function Status8() {
  return (
    <div className="absolute contents left-[1305.5px] top-[347.5px]" data-name="Status">
      <div className="absolute bg-[#48bb78] h-[25px] left-[1305.5px] rounded-[8px] top-[347.5px] w-[65px]" />
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1338px] not-italic text-[14px] text-center text-white top-[350px] translate-x-[-50%] w-[44px]">Online</p>
    </div>
  );
}

function Function3() {
  return (
    <div className="absolute contents leading-[1.4] left-[1045px] not-italic text-[14px] top-[342px]" data-name="Function">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] left-[1045px] text-[#718096] top-[359px] w-[51px]">Projects</p>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] left-[1045px] text-[#2d3748] top-[342px] w-[65.5px]">Executive</p>
    </div>
  );
}

function Name3() {
  return (
    <div className="absolute contents leading-[1.4] left-[375.5px] not-italic text-[14px] top-[342px]" data-name="Name">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] left-[375.5px] text-[#718096] top-[359px] w-[151.5px]">laurent@simmmple.com</p>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] left-[375.5px] text-[#2d3748] top-[342px] w-[107px]">Laurent Michael</p>
    </div>
  );
}

function Image3() {
  return (
    <div className="absolute contents left-[320.5px] top-[340px]" data-name="Image">
      <div className="absolute bg-[#4fd1c5] left-[320.5px] rounded-[12px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] size-[40px] top-[340px]" />
      <div className="absolute inset-[29.55%_81.17%_65.39%_16.46%] mask-alpha mask-intersect mask-no-clip mask-no-repeat mask-position-[-1px_2px] mask-size-[51px_51px]" data-name="Credits to Unsplash.com" style={{ maskImage: `url('${imgCreditsToUnsplashCom}')` }}>
        <img alt="" className="absolute inset-0 max-w-none object-50%-50% object-cover pointer-events-none size-full" src={imgCreditsToUnsplashCom4} />
      </div>
    </div>
  );
}

function Avatar3() {
  return (
    <div className="absolute contents left-[320.5px] top-[340px]" data-name="Avatar">
      <Name3 />
      <Image3 />
    </div>
  );
}

function Content8() {
  return (
    <div className="absolute contents left-[320.5px] top-[340px]" data-name="Content">
      <Edit3 />
      <Employed3 />
      <Status8 />
      <Function3 />
      <Avatar3 />
    </div>
  );
}

function LaurentMichael() {
  return (
    <div className="absolute contents left-[320.5px] top-[340px]" data-name="Laurent Michael">
      <Line6 />
      <Content8 />
    </div>
  );
}

function Line7() {
  return (
    <div className="absolute h-0 left-[320.5px] top-[329px] w-[1555px]" data-name="Line">
      <div className="absolute inset-[-0.5px_-0.03%]">
        <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 1556 1">
          <g id="Line">
            <path d="M0.5 0.5H1555.5" id="Vector 5" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </svg>
      </div>
    </div>
  );
}

function Edit4() {
  return (
    <div className="absolute contents left-[1781px] top-[289px]" data-name="Edit">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[18px] leading-[1.5] left-[1781px] not-italic text-[#718096] text-[12px] top-[289px] w-[23px]">Edit</p>
    </div>
  );
}

function Employed4() {
  return (
    <div className="absolute contents left-[1552px] top-[288.5px]" data-name="Employed">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1579.5px] not-italic text-[#2d3748] text-[14px] text-center top-[288.5px] translate-x-[-50%] w-[55px]">14/06/21</p>
    </div>
  );
}

function Status9() {
  return (
    <div className="absolute contents left-[1305.5px] top-[285.5px]" data-name="Status">
      <div className="absolute bg-[#cbd5e0] h-[25px] left-[1305.5px] rounded-[8px] top-[285.5px] w-[65px]" />
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1338.25px] not-italic text-[14px] text-center text-white top-[288px] translate-x-[-50%] w-[44.5px]">Offline</p>
    </div>
  );
}

function Function4() {
  return (
    <div className="absolute contents leading-[1.4] left-[1045px] not-italic text-[14px] top-[280px]" data-name="Function">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] left-[1045px] text-[#718096] top-[297px] w-[64px]">Developer</p>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] left-[1045px] text-[#2d3748] top-[280px] w-[83.5px]">Programmer</p>
    </div>
  );
}

function Name4() {
  return (
    <div className="absolute contents leading-[1.4] left-[375.5px] not-italic text-[14px] top-[280px]" data-name="Name">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] left-[375.5px] text-[#718096] top-[297px] w-[142px]">alexa@simmmple.com</p>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] left-[375.5px] text-[#2d3748] top-[280px] w-[75px]">Alexa Liras</p>
    </div>
  );
}

function Image4() {
  return (
    <div className="absolute contents left-[320.5px] top-[278px]" data-name="Image">
      <div className="absolute bg-[#4fd1c5] left-[320.5px] rounded-[12px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] size-[40px] top-[278px]" />
      <div className="absolute inset-[24.1%_81.17%_70.84%_16.46%] mask-alpha mask-intersect mask-no-clip mask-no-repeat mask-position-[-1px_2px] mask-size-[51px_51px]" data-name="Credits to Unsplash.com" style={{ maskImage: `url('${imgCreditsToUnsplashCom}')` }}>
        <img alt="" className="absolute inset-0 max-w-none object-50%-50% object-cover pointer-events-none size-full" src={imgCreditsToUnsplashCom5} />
      </div>
    </div>
  );
}

function Avatar4() {
  return (
    <div className="absolute contents left-[320.5px] top-[278px]" data-name="Avatar">
      <Name4 />
      <Image4 />
    </div>
  );
}

function Content9() {
  return (
    <div className="absolute contents left-[320.5px] top-[278px]" data-name="Content">
      <Edit4 />
      <Employed4 />
      <Status9 />
      <Function4 />
      <Avatar4 />
    </div>
  );
}

function AlexaLiras() {
  return (
    <div className="absolute contents left-[320.5px] top-[278px]" data-name="Alexa Liras">
      <Line7 />
      <Content9 />
    </div>
  );
}

function Line8() {
  return (
    <div className="absolute h-0 left-[320.5px] top-[267px] w-[1555px]" data-name="Line">
      <div className="absolute inset-[-0.5px_-0.03%]">
        <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 1556 1">
          <g id="Line">
            <path d="M0.5 0.5H1555.5" id="Vector 5" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </svg>
      </div>
    </div>
  );
}

function Edit5() {
  return (
    <div className="absolute contents left-[1781px] top-[227px]" data-name="Edit">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[18px] leading-[1.5] left-[1781px] not-italic text-[#718096] text-[12px] top-[227px] w-[23px]">Edit</p>
    </div>
  );
}

function Employed5() {
  return (
    <div className="absolute contents left-[1552px] top-[226.5px]" data-name="Employed">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1579.5px] not-italic text-[#2d3748] text-[14px] text-center top-[226.5px] translate-x-[-50%] w-[55px]">14/06/21</p>
    </div>
  );
}

function Status10() {
  return (
    <div className="absolute contents left-[1305.5px] top-[223.5px]" data-name="Status">
      <div className="absolute bg-[#48bb78] h-[25px] left-[1305.5px] rounded-[8px] top-[223.5px] w-[65px]" />
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[1338px] not-italic text-[14px] text-center text-white top-[226px] translate-x-[-50%] w-[44px]">Online</p>
    </div>
  );
}

function Function5() {
  return (
    <div className="absolute contents leading-[1.4] left-[1045px] not-italic text-[14px] top-[218px]" data-name="Function">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] left-[1045px] text-[#718096] top-[235px] w-[79.5px]">Organization</p>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] left-[1045px] text-[#2d3748] top-[218px] w-[58px]">Manager</p>
    </div>
  );
}

function Name5() {
  return (
    <div className="absolute contents leading-[1.4] left-[375.5px] not-italic text-[14px] top-[218px]" data-name="Name">
      <p className="absolute font-['Helvetica:Regular',sans-serif] h-[19.5px] left-[375.5px] text-[#718096] top-[235px] w-[155.5px]">esthera@simmmple.com</p>
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] left-[375.5px] text-[#2d3748] top-[218px] w-[111.5px]">Esthera Jackson</p>
    </div>
  );
}

function Image5() {
  return (
    <div className="absolute contents left-[320.5px] top-[216px]" data-name="Image">
      <div className="absolute bg-[#4fd1c5] left-[320.5px] rounded-[12px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] size-[40px] top-[216px]" />
      <div className="absolute inset-[18.65%_81.17%_76.3%_16.46%] mask-alpha mask-intersect mask-no-clip mask-no-repeat mask-position-[-1px_2px] mask-size-[51px_51px]" data-name="Credits to Unsplash.com" style={{ maskImage: `url('${imgCreditsToUnsplashCom}')` }}>
        <img alt="" className="absolute inset-0 max-w-none object-50%-50% object-cover pointer-events-none size-full" src={imgCreditsToUnsplashCom6} />
      </div>
    </div>
  );
}

function Avatar5() {
  return (
    <div className="absolute contents left-[320.5px] top-[216px]" data-name="Avatar">
      <Name5 />
      <Image5 />
    </div>
  );
}

function Content10() {
  return (
    <div className="absolute contents left-[320.5px] top-[216px]" data-name="Content">
      <Edit5 />
      <Employed5 />
      <Status10 />
      <Function5 />
      <Avatar5 />
    </div>
  );
}

function EstheraJackson() {
  return (
    <div className="absolute contents left-[320.5px] top-[216px]" data-name="Esthera Jackson">
      <Line8 />
      <Content10 />
    </div>
  );
}

function Items1() {
  return (
    <div className="absolute contents left-[320.5px] top-[216px]" data-name="Items">
      <MarkWilson />
      <DanielThomas />
      <FreduardoHill />
      <LaurentMichael />
      <AlexaLiras />
      <EstheraJackson />
    </div>
  );
}

function Lines1() {
  return (
    <div className="absolute h-0 left-[320.5px] top-[205px] w-[1555px]" data-name="Lines">
      <div className="absolute inset-[-0.5px_-0.03%]">
        <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 1556 1">
          <g id="Line">
            <path d="M0.5 0.5H1555.5" id="Vector 5" stroke="var(--stroke-0, #E2E8F0)" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </svg>
      </div>
    </div>
  );
}

function Titles1() {
  return (
    <div className="absolute contents font-['Helvetica:Bold',sans-serif] leading-[1.5] left-[320.5px] not-italic text-[#a0aec0] text-[10px] top-[177.5px]" data-name="Titles">
      <p className="absolute h-[15px] left-[1551.5px] top-[177.5px] w-[56.5px]">EMPLOYED</p>
      <p className="absolute h-[15px] left-[1318.5px] top-[177.5px] w-[39px]">STATUS</p>
      <p className="absolute h-[15px] left-[1045px] top-[177.5px] w-[52px]">FUNCTION</p>
      <p className="absolute h-[15px] left-[320.5px] top-[177.5px] w-[43px]">AUTHOR</p>
    </div>
  );
}

function List1() {
  return (
    <div className="absolute contents left-[320.5px] top-[177.5px]" data-name="List">
      <Items1 />
      <Lines1 />
      <Titles1 />
    </div>
  );
}

function Text1() {
  return (
    <div className="absolute contents left-[319px] top-[129px]" data-name="Text">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[25px] leading-[1.4] left-[319px] not-italic text-[#2d3748] text-[18px] top-[129px] w-[120px]">Authors Table</p>
    </div>
  );
}

function FirstCard() {
  return (
    <div className="absolute contents left-[298px] top-[101px]" data-name="First Card">
      <Background2 />
      <List1 />
      <Text1 />
    </div>
  );
}

function ContentCards() {
  return (
    <div className="absolute contents left-[298px] top-[101px]" data-name="Content Cards">
      <SecondCard />
      <FirstCard />
    </div>
  );
}

function Icon10() {
  return (
    <div className="relative shrink-0 size-[15px]" data-name="Icon">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 15 15">
        <g id="Icon">
          <path d={svgPaths.p15871900} fill="var(--fill-0, #2D3748)" id="Vector" />
        </g>
      </svg>
    </div>
  );
}

function Addon() {
  return (
    <div className="box-border content-stretch flex gap-[5px] h-[20px] items-center overflow-clip px-[6px] py-[4px] relative shrink-0" data-name="Addon">
      <Icon10 />
    </div>
  );
}

function AutoAddedFrame() {
  return (
    <div className="content-stretch flex h-full items-center justify-center overflow-clip relative shrink-0 w-[37.5px]" data-name="Auto-added frame">
      <Addon />
    </div>
  );
}

function MinWidth() {
  return (
    <div className="box-border content-stretch flex items-center justify-center overflow-clip px-[74px] py-0 relative shrink-0" data-name="🔛MinWidth">
      <div className="shrink-0 size-[0.006px]" data-name="Content" />
    </div>
  );
}

function InputFieldText() {
  return (
    <div className="content-stretch flex flex-col items-start justify-center overflow-clip relative shrink-0" data-name="_Input/FieldText">
      <p className="font-['Helvetica:Regular',sans-serif] h-[18px] leading-[1.5] not-italic relative shrink-0 text-[#a0aec0] text-[12px] w-[63.5px]">Type here...</p>
      <MinWidth />
    </div>
  );
}

function InputWithAddons() {
  return (
    <div className="absolute bg-white inset-[1.98%_9.56%_94.55%_80.08%] rounded-[15px]" data-name="_Input/WithAddons">
      <div className="content-stretch flex items-center overflow-clip relative rounded-[inherit] size-full">
        <AutoAddedFrame />
        <InputFieldText />
      </div>
      <div aria-hidden="true" className="absolute border-[0.5px] border-slate-200 border-solid inset-0 pointer-events-none rounded-[15px]" />
    </div>
  );
}

function IonIconPPersonDefault() {
  return (
    <div className="relative shrink-0 size-[12px]" data-name="IONIcon/P/person/default">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 12 12">
        <g id="IONIcon/P/person/default">
          <path d={svgPaths.pc702380} fill="var(--fill-0, #718096)" id="Vector" />
          <path d={svgPaths.p1d294b80} fill="var(--fill-0, #718096)" id="Vector_2" />
        </g>
      </svg>
    </div>
  );
}

function ListItemDefault() {
  return (
    <div className="absolute content-stretch flex gap-[4px] h-[16px] items-center left-[1754.5px] overflow-clip top-[34px] w-[58px]" data-name="List/Item/Default">
      <IonIconPPersonDefault />
      <div className="flex flex-col font-['Helvetica:Bold',sans-serif] h-[12px] justify-center leading-[0] not-italic relative shrink-0 text-[#718096] text-[12px] w-[66.5px]">
        <p className="leading-[1.5]">Sign In</p>
      </div>
    </div>
  );
}

function IonIconSSettingsSharp() {
  return (
    <div className="absolute left-[1831.5px] size-[12px] top-[36.5px]" data-name="IONIcon/S/settings/sharp">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 12 12">
        <g clipPath="url(#clip0_1_571)" id="IONIcon/S/settings/sharp">
          <path d={svgPaths.p18a1c500} fill="var(--fill-0, #718096)" id="Vector" />
        </g>
        <defs>
          <clipPath id="clip0_1_571">
            <rect fill="white" height="12" width="12" />
          </clipPath>
        </defs>
      </svg>
    </div>
  );
}

function IonIconNNotificationsDefault() {
  return (
    <div className="absolute left-[1860.5px] size-[12px] top-[36.5px]" data-name="IONIcon/N/notifications/default">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 12 12">
        <g clipPath="url(#clip0_1_638)" id="IONIcon/N/notifications/default">
          <path d={svgPaths.p1ecf2900} fill="var(--fill-0, #718096)" id="Vector" />
          <path d={svgPaths.p3a621a00} fill="var(--fill-0, #718096)" id="Vector_2" />
        </g>
        <defs>
          <clipPath id="clip0_1_638">
            <rect fill="white" height="12" width="12" />
          </clipPath>
        </defs>
      </svg>
    </div>
  );
}

function Menu1() {
  return (
    <div className="absolute contents left-[1537.5px] top-[22.5px]" data-name="Menu">
      <InputWithAddons />
      <ListItemDefault />
      <IonIconSSettingsSharp />
      <IonIconNNotificationsDefault />
    </div>
  );
}

function BreadcrumbItemPrevious() {
  return (
    <div className="absolute content-stretch flex flex-col gap-[5px] items-start left-[314.5px] top-[31.5px] w-[45px]" data-name="Breadcrumb/Item/Previous">
      <p className="font-['Helvetica:Regular',sans-serif] leading-[1.5] not-italic relative shrink-0 text-[0px] text-[12px] text-black text-nowrap whitespace-pre">
        <span className="text-[#a0aec0]">{`Pages `}</span> <span className="text-[#2d3748]">{`/  Tables`}</span>
      </p>
    </div>
  );
}

function Text2() {
  return (
    <div className="absolute contents left-[314.5px] top-[31.5px]" data-name="Text">
      <p className="absolute font-['Helvetica:Bold',sans-serif] h-[19.5px] leading-[1.4] left-[314.5px] not-italic text-[#2d3748] text-[14px] top-[55px] w-[43.5px]">Tables</p>
      <BreadcrumbItemPrevious />
    </div>
  );
}

function Breadcrumb() {
  return (
    <div className="absolute contents left-[314.5px] top-[22.5px]" data-name="Breadcrumb">
      <Menu1 />
      <Text2 />
    </div>
  );
}

function MainDashboard() {
  return (
    <div className="absolute contents left-px top-0" data-name="Main Dashboard">
      <Background />
      <FooterMenu />
      <ContentCards />
      <Breadcrumb />
    </div>
  );
}

function Group() {
  return (
    <div className="absolute left-[77.5px] mask-alpha mask-intersect mask-no-clip mask-no-repeat mask-position-[-42.5px_8.5px] mask-size-[218px_169.5px] size-[355.5px] top-[586px]" style={{ maskImage: `url('${imgGroup1}')` }}>
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 356 356">
        <g id="Group 1" opacity="0.2">
          <circle cx="177.75" cy="177.75" id="Ellipse 1" r="177.25" stroke="var(--stroke-0, white)" />
          <circle cx="177.75" cy="177.75" id="Ellipse 2" r="153.219" stroke="var(--stroke-0, white)" />
          <circle cx="177.75" cy="177.75" id="Ellipse 3" r="125.872" stroke="var(--stroke-0, white)" />
          <circle cx="177.75" cy="177.75" id="Ellipse 4" r="98.5262" stroke="var(--stroke-0, white)" />
          <circle cx="177.75" cy="177.75" id="Ellipse 5" r="72.0087" stroke="var(--stroke-0, white)" />
          <circle cx="177.75" cy="177.75" id="Ellipse 6" r="47.9773" stroke="var(--stroke-0, white)" />
          <circle cx="177.75" cy="177.75" id="Ellipse 7" r="28.75" stroke="var(--stroke-0, white)" />
        </g>
      </svg>
    </div>
  );
}

function Background3() {
  return (
    <div className="absolute contents left-[35px] top-[594.5px]" data-name="Background">
      <div className="absolute bg-[#4fd1c5] h-[169.5px] left-[35px] rounded-[15px] top-[594.5px] w-[218px]" />
      <Group />
    </div>
  );
}

function ButtonBody() {
  return (
    <div className="content-stretch flex gap-[4px] items-start overflow-clip relative shrink-0" data-name="Button Body">
      <div className="flex flex-col font-['Helvetica:Bold',sans-serif] h-[15px] justify-center leading-[0] not-italic relative shrink-0 text-[#2d3748] text-[10px] text-center w-[87.5px]">
        <p className="leading-[1.5]">DOCUMENTATION</p>
      </div>
    </div>
  );
}

function HeightStructure() {
  return (
    <div className="content-stretch flex h-[24px] items-center relative shrink-0" data-name="Height Structure">
      <ButtonBody />
    </div>
  );
}

function MinWidth1() {
  return (
    <div className="box-border content-stretch flex items-start overflow-clip px-[12px] py-0 relative shrink-0" data-name="🔛MinWidth">
      <div className="bg-[#c4c4c4] shrink-0 size-[0.006px]" data-name="Content" />
    </div>
  );
}

function WidthStructure() {
  return (
    <div className="content-stretch flex flex-col items-center justify-center overflow-clip relative shrink-0" data-name="Width Structure">
      <HeightStructure />
      <MinWidth1 />
    </div>
  );
}

function ButtonBase() {
  return (
    <div className="absolute bg-white box-border content-stretch flex inset-[62.71%_87.66%_34.21%_2.66%] items-center justify-center px-[8px] py-0 rounded-[12px]" data-name="_Button/Base">
      <WidthStructure />
    </div>
  );
}

function Text3() {
  return (
    <div className="absolute contents leading-[0] left-[51px] not-italic text-white top-[667px]" data-name="Text">
      <div className="absolute flex flex-col font-['Helvetica:Regular',sans-serif] h-[18px] justify-center left-[51.5px] text-[12px] top-[695.5px] translate-y-[-50%] w-[121px]">
        <p className="leading-[1.5]">Please check our docs</p>
      </div>
      <div className="absolute flex flex-col font-['Helvetica:Bold',sans-serif] h-[19.5px] justify-center left-[51px] text-[14px] top-[676.75px] translate-y-[-50%] w-[76px]">
        <p className="leading-[1.4]">Need help?</p>
      </div>
    </div>
  );
}

function IonIconHHelpCircle() {
  return (
    <div className="absolute left-[57px] size-[24px] top-[616px]" data-name="IONIcon/H/help/circle">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 24 24">
        <g id="IONIcon/H/help/circle">
          <path d={svgPaths.p29ff8700} fill="var(--fill-0, #4FD1C5)" id="Vector" />
        </g>
      </svg>
    </div>
  );
}

function Icon11() {
  return (
    <div className="absolute contents left-[51.5px] top-[610.5px]" data-name="Icon">
      <div className="absolute bg-white left-[51.5px] rounded-[12px] size-[35px] top-[610.5px]" />
      <IonIconHHelpCircle />
    </div>
  );
}

function Group1() {
  return (
    <div className="absolute contents left-[35px] top-[594.5px]">
      <Background3 />
      <ButtonBase />
      <Text3 />
      <Icon11 />
    </div>
  );
}

function NeedHelp() {
  return (
    <div className="absolute contents left-[18px] top-[594.5px]" data-name="Need Help">
      <div className="absolute bg-[#f8f9fa] h-[214.5px] left-[18px] top-[845px] w-[246.5px]" />
      <Group1 />
    </div>
  );
}

function IonIconRRocketSharp() {
  return (
    <div className="absolute left-[55px] size-[15px] top-[502.5px]" data-name="IONIcon/R/rocket/sharp">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 15 15">
        <g clipPath="url(#clip0_1_646)" id="IONIcon/R/rocket/sharp">
          <path d={svgPaths.p61c4780} fill="var(--fill-0, #4FD1C5)" id="Vector" />
          <path d={svgPaths.pc0c1870} fill="var(--fill-0, #4FD1C5)" id="Vector_2" />
        </g>
        <defs>
          <clipPath id="clip0_1_646">
            <rect fill="white" height="15" width="15" />
          </clipPath>
        </defs>
      </svg>
    </div>
  );
}

function SignUp() {
  return (
    <div className="absolute contents left-[47.5px] top-[495px]" data-name="Sign Up">
      <div className="absolute bg-white left-[47.5px] rounded-[12px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] size-[30px] top-[495px]" />
      <IonIconRRocketSharp />
      <div className="absolute flex flex-col font-['Helvetica:Bold',sans-serif] h-[18px] justify-center leading-[0] left-[89.5px] not-italic text-[#a0aec0] text-[12px] top-[510px] translate-y-[-50%] w-[45.5px]">
        <p className="leading-[1.5]">Sign Up</p>
      </div>
    </div>
  );
}

function IonIconDDocumentDefault() {
  return (
    <div className="absolute left-[55px] size-[15px] top-[448.5px]" data-name="IONIcon/D/document/default">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 15 15">
        <g id="IONIcon/D/document/default">
          <path d={svgPaths.p1b4209c0} fill="var(--fill-0, #4FD1C5)" id="Vector" />
          <path d={svgPaths.p3ae50c80} fill="var(--fill-0, #4FD1C5)" id="Vector_2" />
        </g>
      </svg>
    </div>
  );
}

function SignIn() {
  return (
    <div className="absolute contents left-[47.5px] top-[441px]" data-name="Sign In">
      <div className="absolute bg-white left-[47.5px] rounded-[12px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] size-[30px] top-[441px]" />
      <IonIconDDocumentDefault />
      <div className="absolute flex flex-col font-['Helvetica:Bold',sans-serif] h-[18px] justify-center leading-[0] left-[89.5px] not-italic text-[#a0aec0] text-[12px] top-[456px] translate-y-[-50%] w-[40.5px]">
        <p className="leading-[1.5]">Sign In</p>
      </div>
    </div>
  );
}

function IonIconPPersonDefault1() {
  return (
    <div className="absolute left-[55px] size-[15px] top-[394.5px]" data-name="IONIcon/P/person/default">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 15 15">
        <g id="IONIcon/P/person/default">
          <path d={svgPaths.pc31f100} fill="var(--fill-0, #4FD1C5)" id="Vector" />
          <path d={svgPaths.pb496600} fill="var(--fill-0, #4FD1C5)" id="Vector_2" />
        </g>
      </svg>
    </div>
  );
}

function Profile() {
  return (
    <div className="absolute contents left-[47.5px] top-[387px]" data-name="Profile">
      <div className="absolute bg-white left-[47.5px] rounded-[12px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] size-[30px] top-[387px]" />
      <IonIconPPersonDefault1 />
      <div className="absolute flex flex-col font-['Helvetica:Bold',sans-serif] h-[18px] justify-center leading-[0] left-[89.5px] not-italic text-[#a0aec0] text-[12px] top-[402px] translate-y-[-50%] w-[37.5px]">
        <p className="leading-[1.5]">Profile</p>
      </div>
    </div>
  );
}

function IonIconBBuildDefault() {
  return (
    <div className="absolute left-[55px] size-[15px] top-[298.5px]" data-name="IONIcon/B/build/default">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 15 15">
        <g clipPath="url(#clip0_1_568)" id="IONIcon/B/build/default">
          <path d={svgPaths.p2051c770} fill="var(--fill-0, #4FD1C5)" id="Vector" />
        </g>
        <defs>
          <clipPath id="clip0_1_568">
            <rect fill="white" height="15" width="15" />
          </clipPath>
        </defs>
      </svg>
    </div>
  );
}

function Rtl() {
  return (
    <div className="absolute contents left-[47.5px] top-[291px]" data-name="RTL">
      <div className="absolute bg-white left-[47.5px] rounded-[12px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] size-[30px] top-[291px]" />
      <IonIconBBuildDefault />
      <div className="absolute flex flex-col font-['Helvetica:Bold',sans-serif] h-[18px] justify-center leading-[0] left-[89.5px] not-italic text-[#a0aec0] text-[12px] top-[306px] translate-y-[-50%] w-[23.5px]">
        <p className="leading-[1.5]">RTL</p>
      </div>
    </div>
  );
}

function IonIconCCardDefault() {
  return (
    <div className="absolute left-[55px] size-[15px] top-[244.5px]" data-name="IONIcon/C/card/default">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 15 15">
        <g id="IONIcon/C/card/default">
          <path d={svgPaths.p123c1000} fill="var(--fill-0, #4FD1C5)" id="Vector" />
          <path d={svgPaths.p379a1a00} fill="var(--fill-0, #4FD1C5)" id="Vector_2" />
        </g>
      </svg>
    </div>
  );
}

function Billing() {
  return (
    <div className="absolute contents left-[47.5px] top-[237px]" data-name="Billing">
      <div className="absolute bg-white left-[47.5px] rounded-[12px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] size-[30px] top-[237px]" />
      <IonIconCCardDefault />
      <div className="absolute flex flex-col font-['Helvetica:Bold',sans-serif] h-[18px] justify-center leading-[0] left-[89.5px] not-italic text-[#a0aec0] text-[12px] top-[252px] translate-y-[-50%] w-[37px]">
        <p className="leading-[1.5]">Billing</p>
      </div>
    </div>
  );
}

function IonIconSStatsChart() {
  return (
    <div className="absolute left-[55px] size-[15px] top-[190.5px]" data-name="IONIcon/S/stats/chart">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 15 15">
        <g id="IONIcon/S/stats/chart">
          <path d={svgPaths.p2f34f400} fill="var(--fill-0, #4FD1C5)" id="Vector" />
          <path d={svgPaths.p11144100} fill="var(--fill-0, #4FD1C5)" id="Vector_2" />
          <path d={svgPaths.p34034e00} fill="var(--fill-0, #4FD1C5)" id="Vector_3" />
          <path d={svgPaths.p2e482a00} fill="var(--fill-0, #4FD1C5)" id="Vector_4" />
        </g>
      </svg>
    </div>
  );
}

function Tables() {
  return (
    <div className="absolute contents left-[47.5px] top-[183px]" data-name="Tables">
      <div className="absolute bg-white left-[47.5px] rounded-[12px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] size-[30px] top-[183px]" />
      <IonIconSStatsChart />
      <div className="absolute flex flex-col font-['Helvetica:Bold',sans-serif] h-[18px] justify-center leading-[0] left-[89.5px] not-italic text-[#a0aec0] text-[12px] top-[198px] translate-y-[-50%] w-[37.5px]">
        <p className="leading-[1.5]">Tables</p>
      </div>
    </div>
  );
}

function IonIconHHomeDefault() {
  return (
    <div className="absolute left-[55px] size-[15px] top-[136.5px]" data-name="IONIcon/H/home/default">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 15 15">
        <g clipPath="url(#clip0_1_564)" id="IONIcon/H/home/default">
          <path d={svgPaths.p8c75c00} fill="var(--fill-0, white)" id="Vector" />
          <path d={svgPaths.p3e46f900} fill="var(--fill-0, white)" id="Vector_2" />
        </g>
        <defs>
          <clipPath id="clip0_1_564">
            <rect fill="white" height="15" width="15" />
          </clipPath>
        </defs>
      </svg>
    </div>
  );
}

function Dashboard() {
  return (
    <div className="absolute contents left-[31.5px] top-[117px]" data-name="Dashboard">
      <div className="absolute bg-white h-[54px] left-[31.5px] rounded-[15px] shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)] top-[117px] w-[219.5px]" />
      <div className="absolute bg-[#4fd1c5] left-[47.5px] rounded-[12px] size-[30px] top-[129px]" />
      <IonIconHHomeDefault />
      <div className="absolute flex flex-col font-['Helvetica:Bold',sans-serif] h-[18px] justify-center leading-[0] left-[89.5px] not-italic text-[#2d3748] text-[12px] top-[144px] translate-y-[-50%] w-[63px]">
        <p className="leading-[1.5]">Dashboard</p>
      </div>
    </div>
  );
}

function Menu2() {
  return (
    <div className="absolute contents left-[31.5px] top-[117px]" data-name="Menu">
      <SignUp />
      <SignIn />
      <Profile />
      <div className="absolute flex flex-col font-['Helvetica:Bold',sans-serif] h-[18px] justify-center leading-[0] left-[47.5px] not-italic text-[#2d3748] text-[12px] top-[354px] translate-y-[-50%] w-[105px]">
        <p className="leading-[1.5]">ACCOUNT PAGES</p>
      </div>
      <Rtl />
      <Billing />
      <Tables />
      <Dashboard />
    </div>
  );
}

function Icon12() {
  return (
    <div className="absolute inset-[3.87%_96.67%_94.2%_2.19%]" data-name="icon">
      <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 22 22">
        <g id="icon">
          <path d={svgPaths.p111829d0} fill="var(--fill-0, #2D3748)" id="Shape" />
          <path d={svgPaths.pd181c00} fill="var(--fill-0, #2D3748)" id="Path" />
          <path d={svgPaths.p2195f500} fill="var(--fill-0, #2D3748)" id="Path_2" />
          <path d={svgPaths.p1c133400} fill="var(--fill-0, #2D3748)" id="Path_3" />
        </g>
      </svg>
    </div>
  );
}

function LogoCreativeTimBlack() {
  return (
    <div className="absolute contents inset-[3.87%_96.67%_94.2%_2.19%]" data-name="logo-creative-tim-black">
      <Icon12 />
    </div>
  );
}

function Logo() {
  return (
    <div className="absolute contents left-[42px] top-[44px]" data-name="Logo">
      <div className="absolute flex flex-col font-['Helvetica:Bold',sans-serif] justify-center leading-[0] left-[76px] not-italic text-[#2d3748] text-[14px] text-nowrap top-[56.5px] translate-y-[-50%]">
        <p className="leading-[1.5] whitespace-pre">PURITY UI DASHBOARD</p>
      </div>
      <LogoCreativeTimBlack />
    </div>
  );
}

function Sidebar() {
  return (
    <div className="absolute contents left-[18px] top-[44px]" data-name="Sidebar">
      <NeedHelp />
      <Menu2 />
      <Logo />
      <div className="absolute h-0 left-[24.5px] top-[94.5px] w-[233.25px]">
        <div className="absolute bottom-[-0.5px] left-0 right-0 top-[-0.5px]">
          <svg className="block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 234 1">
            <path d="M0 0.5H233.25" id="Vector 6" stroke="url(#paint0_linear_1_536)" />
            <defs>
              <linearGradient gradientUnits="userSpaceOnUse" id="paint0_linear_1_536" x1="0" x2="231" y1="0.5" y2="0.5">
                <stop stopColor="#E0E1E2" stopOpacity="0" />
                <stop offset="0.5" stopColor="#E0E1E2" />
                <stop offset="1" stopColor="#E0E1E2" stopOpacity="0.15625" />
              </linearGradient>
            </defs>
          </svg>
        </div>
      </div>
    </div>
  );
}

function NewDesign() {
  return (
    <div className="absolute contents left-px top-0" data-name="New Design">
      <MainDashboard />
      <Sidebar />
    </div>
  );
}

export default function TablesScreen() {
  return (
    <div className="relative size-full" data-name="Tables Screen">
      <NewDesign />
    </div>
  );
}