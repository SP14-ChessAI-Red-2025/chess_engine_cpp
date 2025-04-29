import React from 'react';
import './CoverPage.css'; // Create a CSS file for styling if needed

const imageBase = '/website/images/';
const docBase = '/website/documents/';

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
        <p>Instructor: Sharon Perry</p>
      </div>

      <div className="summary-section">
        <h3>Project Summary</h3>
        <p className="project-summary">
          This project is a chess AI engine that leverages advanced algorithms and neural networks to evaluate board states and make intelligent moves. It includes a web-based interface for users to play against the AI or watch AI vs AI games.
        </p>
      </div>

      <div className="links-section">
        <h3>Navigation Links</h3>
        <ul>
          <li>
            <a href={`${docBase}report.pdf`} target="_blank" rel="noopener noreferrer">
              Project Report (PDF)
            </a>
          </li>
          <li>
            <a href={`${docBase}final_presentation.mov`} target="_blank" rel="noopener noreferrer">
              Final Presentation Video - MP4
            </a>
          </li>
          <li>
            <a href="https://www.youtube.com/watch?v=rnROCtE1tSg" target="_blank" rel="noopener noreferrer">
            Final Presentation Video - Youtube
            </a>
          </li>
          <li>
            <a href="https://www.youtube.com/watch?v=m0rQI1WBixc" target="_blank" rel="noopener noreferrer">
              Demo - Youtube
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