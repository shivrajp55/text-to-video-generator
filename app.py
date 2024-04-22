import streamlit as st
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2

# Function to generate and display video
def generate_and_display_video(prompt, video_quality, video_duration_seconds):
    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Optimize for GPU memory
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    # Generate video frames
    num_frames = video_duration_seconds * 10
    video_frames = pipe(prompt, negative_prompt=video_quality, num_inference_steps=25, num_frames=num_frames).frames
    video_frames = video_frames.reshape((num_frames, 256, 256, 3))

    # Export video
    video_path = export_to_video(video_frames)

    # Display video
    video = imageio.mimread(video_path)
    fig = plt.figure(figsize=(4.2, 4.2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    mov = []
    for i in range(len(video)):
        img = plt.imshow(video[i], animated=True)
        plt.axis('off')
        mov.append([img])
    anime = animation.ArtistAnimation(fig, mov, interval=100, repeat_delay=1000)
    plt.close()
    return anime.to_html5_video()


# Navbar
def navbar():
    st.markdown(
        """
       <nav>
            <div class="navbar">
              <div class="logo"><a href="#">Text-to-Video</a></div>
               <ul class="menu">
                  <li><a href="#Home">Home</a></li>
                  <li><a href="#Feedback1">Generate</a></li>
                  <li><a href="#About">About</a></li>
                  <li><a href="#Contact">Contact</a></li>
               </ul>
            </div>
       </nav>
        """,
        unsafe_allow_html=True,
    )

# Home Section
def home_section():
    st.markdown(
        """
        <content id="Home" class="home_section">
             <section>
                 <h1>Welcome to Text to Video Generator</h1>
                <img class="fix" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQqCMocoH5s23Po536cGhckVC_I2_yDhlhSxP5pipweoA&s" alt="#" id ="Feedback1">
             </section>
        </content>
        """,
        unsafe_allow_html=True,
    )

# Feedback Section
# HTML code for the form

# About Section
def about_section():
    st.markdown(
        """
        <content id="About">
              <div class="container">
                  <h2 id="about1">About Section</h2>
                  <div id="about2">
                     <h3>About Text-to-Video Generator</h3>
                     <p>Welcome to our Text-to-Video Generator! Our project utilizes state-of-the-art Diffusion Models to generate captivating videos from text prompts.</p>
                     <h3>How It Works</h3>
                     <p>Using cutting-edge Diffusion Models, our system translates textual prompts into stunning visual
                        representations. By leveraging the capabilities of Diffusion Models, which excel in understanding and
                        generating high-quality images, we're able to create seamless transitions, vivid scenes, and captivating
                       narratives.</p>
                      <h3>Key Features</h3>
                      <ul>
                           <li>Interactive Interface: Our user-friendly interface allows you to input text prompts easily and select
                               the desired video quality and duration.</li>
                           <li>High-Quality Output: Experience videos in stunning high definition, crafted with attention to detail to
                               ensure a visually engaging experience.</li>
                           <li>Customization: Tailor your videos to suit your preferences by adjusting parameters such as video
                               quality.</li>
                       </ul>
                   </div>
               </div>
        </content>

        """,
        unsafe_allow_html=True,
    )

# Contact Section
def contact_section():
    st.markdown(
        """
       <content id="Contact">
            <sect1 class="contact-page-sec">
                <div class="container1">
                    <div class="row">
                         <div class="col-md-4">
                             <div class="contact-info">
                                <div class="contact-info-item">
                                   <div class="contact-info-icon">
                                       <i class="fas fa-map-marked"></i>
                                   </div>
                                  <div class="contact-info-text">
                                     <h2>Address</h2>
                                     <span>1215 Lorem Ipsum, Ch 176080</span>
                                     <span>Chandigarh, INDIA</span>
                                   </div>
                                </div>
                           </div>
                       </div>
              <div class="col-md-4">
                       <div class="contact-info">
                           <div class="contact-info-item">
                               <div class="contact-info-icon">
                                  <i class="fas fa-envelope"></i>
                                </div>
                              <div class="contact-info-text">
                                  <h2>E-mail</h2>
                                  <span>info@LoremIpsum.com</span>
                                  <span>yourmail@gmail.com</span>
                              </div>
                           </div>
                        </div>
               </div>
          <div class="col-md-4">
                   <div class="contact-info">
                      <div class="contact-info-item">
                           <div class="contact-info-icon">
                              <i class="fas fa-clock"></i>
                           </div>
                           <!-- Add your timing information here if needed -->
                       </div>
                   </div>
                 </div>
          </div>
          <div class="row">
              <div class="col-md-8">
                  <div class="contact-page-form" method="post">
                      <h2 id="contact1">Contact Us</h2>
                    <form action="https://formspree.io/f/mzblybbv" method="post">
                        <div class="row">
                             <div class="col-md-6 col-sm-6 col-xs-12">
                             <div class="single-input-field">
                             <input type="text" placeholder="Your Name" name="name" required />
                        </div>
                   </div>
                   <div class="col-md-6 col-sm-6 col-xs-12">
                         <div class="single-input-field">
                             <input type="email" placeholder="E-mail" name="email" required />
                         </div>
                   </div>
                   <div class="col-md-6 col-sm-6 col-xs-12">
                        <div class="single-input-field">
                            <input type="text" placeholder="Phone Number" name="phone" required />
                        </div>
                  </div>
                  <div class="col-md-12 message-input">
                        <div class="single-input-field">
                            <textarea placeholder="Write Your Message" name="message" required></textarea>
                        </div>
                  </div>
                  <div class="col-md-12 single-input-fieldsbtn">
                      <input type="submit" value="Send Now" />
                  </div>
                </div>
              </form>
            </div>
          </div>
             <div class="col-md-4">
                 <!-- Add any additional content here -->
             </div>
            </div>
          </div>
         </sect1>
      </content>

       <div class="footer">
      <div id="footer1">Copyright &copy; 2024 savv . All rights reserved.</div>
       privacy policy
     </div>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(page_title="Text-to-Video Generator")

    # CSS styling
    st.markdown(
        """
        <style>
    .st-emotion-cache-18ni7ap {
            position: fixed;
            top: 0px;
            left: 0px;
            right: 0px;
            height: 2.875rem;
            background: rgb(255, 255, 255);
            outline: none;
            z-index: 999990;
            display: block;
            visibility: hidden;
    }
    
    .st-emotion-cache-1aege4m {
           width: 704px;
           /* position: relative; */
    }
    .st-emotion-cache-gh2jqd {
          padding-left: 1rem;
          background: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUdrVEQSm2XgqtYGWqx0rXROnqeojZmI5eZtDPFm9h1A&s);
          background-size: cover;
    }
    .st-emotion-cache-ue6h4q {
         font-size: 14px;
         color: lawngreen;
         display: flex;
         visibility: visible;
         margin-bottom: 0.25rem;
         height: auto;
         min-height: 1.5rem;
         vertical-align: middle;
         flex-direction: row;
         -webkit-box-align: center;
         align-items: center;
    }
    


    body {
          font-family: Arial, sans-serif;
          color: #000;
          margin: 0;
          padding: 0;
          background: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUdrVEQSm2XgqtYGWqx0rXROnqeojZmI5eZtDPFm9h1A&s');
          background-size: cover;
    }
    h1{
      color: white;
    }
   
 
    * {
         margin: 0;
         padding: 0;
         box-sizing: border-box;
       }
    nav {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background: #2980b9;
    box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
    z-index: 1000;
   }

    .navbar {
         display: flex;
         align-items: center;
         justify-content: space-between;
         max-width: 90%;
         margin: auto;
         height: 50px;
    }

        .logo a {
            color: #fff;
            font-size: 24px;
            font-weight: 600;
            text-decoration: none;
            padding: 10px;
        }


     .menu {
         display: flex;
         list-style-type: none;
      }

        .menu li {
            margin: 0 15px;
        }

      .menu li a {
         color: #fff;
         text-decoration: none;
         font-size: 16px;
      }
       content {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 75px; /* Adjust as needed */
            padding: 20px;
       }

    .fix {
        max-width: 100%;
        height: auto;
    }

    .container {
        max-width: 800px;
        margin: 20px auto;
    }


    #about1 {
            text-align: center;
            background-color: black;
            color: white;
            padding: 10px 0px;
            margin-top: -113px;
     }

       #about2 {
         padding: 20px;
      }

       #about2 h3 {
          margin-top: 20px;
       }

       #about2 p {
         margin-bottom: 20px;
         line-height: 1.6;
       }
       sect1 {
          padding: 47px 0;
          min-height: 67vh;
          background: linear-gradient(45deg, aqua, blue);
          width: 100%;
          margin-top: -95px;
      }


      sect1 {
        padding: 47px 0;
        min-height: 67vh;
        background: linear-gradient(45deg, rgb(171 9 136), #ff1f00c7);
        width: 100%;
        margin-top: -118px;
      }


    .contact-page-sec .container1 {
         max-width: 800px;
         margin: 0 auto;
         padding: 20px;
    }

    .contact-info {
         width: 100%;
         text-align: center;
         margin-bottom: -146px;
     }
    .contact-info {
           width: 100%;
           text-align: center;
           margin-bottom: -185px;
          text-align: center;
    }

    .contact-info-icon {
            margin-bottom: 15px;
      }

    .contact-info-item {
               background: #071c34;
               padding: 30px 0px;
               visibility: hidden;
         }

    .contact-info-text h2 {
             color: white;
             margin-bottom: 5px;
    }

    .contact-info-text span {
          color: white;
          display: block;
          margin-bottom: 5px;
    }

    .contact-page-form h2 {
            color: white;
            text-transform: capitalize;
            font-size: 22px;
            font-weight: 700;
            margin-top: 20px;
    }
    .contact-page-form h2 {
         color: white;
         text-transform: capitalize;
         font-size: 22px;
         font-weight: 700;
         margin-top: 20px;
         text-align: center;
    }

    .contact-page-form .row {
          margin-top: 20px;
    }

    .single-input-field input, .single-input-field textarea {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
    }

    .single-input-fieldsbtn input[type="submit"] {
         background: #fda40b;
         color: #fff;
         font-weight: bold;
         cursor: pointer;
         transition: background-color 0.3s;
         border: none;
         display: flex;
         justify-content: center;
     }
        .single-input-fieldsbtn input[type="submit"] {
             background: #fda40b;
             color: #fff;
             font-weight: bold;
             cursor: pointer;
             transition: background-color 0.3s;
             border: none;
             border-radius: 10px;
             height: 37px;
             width: 85px;
             display: flex;
            justify-content: center;
        }

    .single-input-fieldsbtn input[type="submit"]:hover {
          background-color: #071c34;
    }
    .single-input-fieldsbtn {
      display: flex;
      justify-content: center;
    }

     /* Responsive Styles */
    @media only screen and (max-width: 768px) {
        .navbar {
           flex-direction: column;
           align-items: center;
        }

       .logo a {
           font-size: 20px;
        }
        

       .menu {
         margin-top: 20px;
       }

       .menu li {
         margin: 10px 0;
       }

       content {
         margin-top: 100px;
       }

    .container {
      max-width: 90%;
    }

    .fix {
      max-width: 100%;
      height: auto;
    }

     .single-input-field input,.single-input-field textarea {
         width: calc(100% - 20px);
       }
    }

    @media only screen and (max-width: 576px) {
          .menu li {
            margin: 5px 0;
          }
      }
    @media (min-width: 576px){
         .st-emotion-cache-gh2jqd {
                 padding-left: 1rem;
                 padding-right: 1rem;
                 background: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUdrVEQSm2XgqtYGWqx0rXROnqeojZmI5eZtDPFm9h1A&s);
                 background-size: cover;
             } 
    }
      

    #Feedback {
         background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUdrVEQSm2XgqtYGWqx0rXROnqeojZmI5eZtDPFm9h1A&s');
         background-size: cover;
         margin: 0;
         padding: 0;
    }
      .bg-image-container {
                background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUdrVEQSm2XgqtYGWqx0rXROnqeojZmI5eZtDPFm9h1A&s');
                background-size: cover;
                padding: 50px 0; /* Adjust padding as needed */
                text-align: center;
       }

      #generate {
          margin-top: 15px;
       }

      .form-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
      }

        .form-container h2 {
            color: #fff;
            text-align: center;
            margin-bottom: 20px;
        }

        .form-container label {
            color: #fff;
        }

      .form-container input[type="text"],.form-container select, .form-container input[type="number"], .form-container button {
            width: 100%;
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            border: none;
           font-size: 16px;
      }

         .form-container button {
             background-color: #ff6600;
             color: #ffffff;
             font-weight: bold;
             cursor: pointer;
             transition: background-color 0.3s;
          }

          

          .form-container button:hover {
               background-color: #ff8533;
          }

          .smaller-text {
            font-size: 14px;
          }

        @media only screen and (max-width: 768px) {
          .form-container input[type="text"],.form-container select,.form-container input[type="number"],.form-container button {
                width: calc(100% - 20px);
           }
        }
        .footer {
           /* display: flex; */
           justify-content: center;
           text-align: center;
           color: white;
           background-color: black;
           width: 100%;
           height: 92px;
           font: message-box;
        }
        .footer {
          display: flex;
          justify-content: center;
          text-align: center;
          color: white;
          background-color: black;
          width: 100%;
          height: 57px;
          font: message-box;
          /* margin-top: 10px; */
          padding: 18px 5px;
        }
        .st-emotion-cache-1n76uvr {
            width: 704px;
            position: relative;
            display: flex;
            flex: 1 1 0%;
            flex-direction: column;
            gap: 1rem;
            margin-top: -145px;
        }


        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render navbar
    navbar()
    # Render sections
    home_section()

    
    st.markdown(
        """
        <script>
        function scrollToSection(sectionId) {
            var section = document.getElementById(sectionId);
            section.scrollIntoView({ behavior: 'smooth' });
        }
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<a name='Feedback'></a>", unsafe_allow_html=True)
    # Form to generate video
    st.title("Video Generation")
    prompt = st.text_input('**Enter prompt:**', key="prompt")
    default_video_quality = "Low Quality" if prompt else "High Quality"
    video_quality = st.selectbox('**Select Video Quality:**', ['High Quality', 'Low Quality'], index=0 if default_video_quality == "High Quality" else 1, key="video_quality")
    video_duration_seconds = st.number_input('**Video duration (seconds):**', min_value=1, max_value=10, value=3, key="video_duration_seconds")

    # Anchor tag for scrolling
    
    if st.button('Generate Video'):
          with st.spinner('Generating video...'):
              video_display = generate_and_display_video(prompt, video_quality, video_duration_seconds)
              st.write(video_display, unsafe_allow_html=True)
    about_section()
    contact_section()
if __name__ == "__main__":
    main()
