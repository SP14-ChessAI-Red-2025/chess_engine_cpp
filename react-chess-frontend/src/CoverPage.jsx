import React from 'react';
import './CoverPage.css'; // Create a CSS file for styling if needed

const imageBase = '/website/images/';

const CoverPage = ({ onProceed }) => {
  return (
    <div className="cover-page">
      <h1 className="project-title">SP-14 - Chess AI - Red</h1>

      <div className="team-section">
        <h2>Team Members</h2>
        <div className="team-members">
          <div className="member">
            <img src={`${imageBase}carter.png`} alt="Carter Barnard" className="bio-pic" />
            <p>Carter Barnard</p>
          </div>
          <div className="member">
            <img src={`${imageBase}andy.png`} alt="Andy Kenmoe" className="bio-pic" />
            <p>Andy Kenmoe</p>
          </div>
          <div className="member">
            <img src={`${imageBase}bailey.png`} alt="Bailey Sweeney" className="bio-pic" />
            <p>Bailey Sweeney</p>
          </div>
          <div className="member">
            <img src={`${imageBase}alec.png`} alt="Alec Walsh" className="bio-pic" />
            <p>Alec Walsh</p>
          </div>
        </div>
      </div>

      <div className="course-info">
        <p>Course: CS4850</p>
        <p>Semester: Spring 2025</p>
      </div>

      <div className="links-section">
        <h3>Navigation Links</h3>
        <ul>
          <li>
            <a href="/documents/report.pdf" target="_blank" rel="noopener noreferrer">
              Project Report (PDF)
            </a>
          </li>
          <li>
            <a href="/documents/final_presentation.mp4" target="_blank" rel="noopener noreferrer">
              Final Presentation Video
            </a>
          </li>
          <li>
            <a href="https://github.com/sp14-chessai-red-2025" target="_blank" rel="noopener noreferrer">
              GitHub Repository
            </a>
          </li>
        </ul>
      </div>

      <button className="proceed-button" onClick={onProceed}>
        Optional - Proceed to the Web-based Game (Depth 4)
      </button>
    </div>
  );
};

export default CoverPage;